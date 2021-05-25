"""
Module defining `Flory-Huggins free energies
<https://en.wikipedia.org/wiki/Flory–Huggins_solution_theory>`_.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>    
"""

import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from pde.tools.numba import jit, nb
from pde.tools.typing import NumberOrArray

from .base import FreeEnergyBase, FreeEnergyNComponentsBase


def xlogx_scalar(x: float) -> float:
    r"""calculates :math:`x \log(x)`, including the corner case x == 0

    Args:
        x (float): The argument

    Returns:
        float: The result
    """
    if x < 0:
        return np.nan
    elif x == 0:
        return 0
    else:
        return x * np.log(x)  # type: ignore


# vectorize the function above
xlogx: Callable[[NumberOrArray], NumberOrArray] = np.vectorize(xlogx_scalar, otypes="d")


class FloryHugginsNComponents(FreeEnergyNComponentsBase):
    r"""Flory-Huggins free energy for arbitrary number of components


    The local free energy density for :math:`N` components described by their volume
    fractions :math:`\phi_i` has the form

    .. math ::
        f(\phi) = \sum_{i=0}^{N-1} \frac{\phi_i}{\nu_i} \ln(\phi_i) +
            \sum_{i,j=0}^{N-1} \frac{\chi_{ij}}{2} \phi_i \phi_j +
            \sum_{i=0}^{N-1} w_i \phi_i

    where :math:`\nu_i` are the relative molecular volumes, :math:`\chi_{ij}` is the
    Flory interaction parameter matrix, and :math:`\alpha_i` determine the internal
    energies, which can affect chemical reactions. Note that the Flory matrix must be
    symmetric, :math:`\chi_{ij} = \chi_{ji}`, with vanishing elements on the diagonal,
    :math:`\chi_{ii} = 0`.

    Since we assume an incompressible system, only :math:`N - 1` components are
    independent. We thus only describe the dynamics of the first :math:`N - 2`
    components, using :math:`\phi_{N-1} = 1 - \sum_{i=0}^{N-2} \phi_i`. Consequently,
    the chemical potentials used in the description are exchange chemical potentials
    :math:`\bar\mu_i = \mu_i - \mu_{N-1}` describing the difference to the chemical
    potential of the removed component.
    """

    variable_bounds_default = (1e-8, 1 - 1e-8)
    explicit_time_dependence = False

    sizes: np.ndarray
    _chis: np.ndarray
    _internal_energies: np.ndarray

    def __init__(
        self,
        num_comp: int,
        chis: NumberOrArray = 0,
        sizes: NumberOrArray = 1,
        internal_energies: NumberOrArray = 0,
        *,
        variables: Sequence[str] = None,
        **kwargs,
    ):
        r"""
        Args:
            num_comp (int):
                Number of components described by this free energy. The number
                of independent components will be one less
            chi (`numpy.ndarray` or float):
                Flory-interaction parameters :math:`\chi_{ij}` with shape
                `(num_comp, num_comp)`. Alternatively, a float value
                may be given, which then determines all off-diagonal entries.
            size (`numpy.ndarray` or float):
                Array of shape `(num_comp,)` determining the relative molecular
                volumes :math:`\nu_i`. Float values are broadcasted to full
                arrays.
            internal_energies (`numpy.ndarray` or float):
                Array with shape `(num_comp,)` setting the internal energies
                :math:`w_i`. Float values are broadcasted to full arrays.
            variables (list):
                The name of the variables in the free energy. If omitted, they will be
                named `phi#`, where # is a integer between 1 and num_comp - 1.
        """
        super().__init__()

        self.num_comp = num_comp
        self.dim = num_comp - 1
        self.sizes = np.broadcast_to(sizes, (num_comp,))

        self._internal_energies = np.zeros(num_comp)  # initialize for calculation
        self.chis = chis  # type: ignore
        self.internal_energies = internal_energies  # type: ignore
        # the last two assignments also calculate the reduced chis and internal_energies

        if "slopes" in kwargs:
            if np.allclose(self.internal_energies, 0):
                # renamed parameter on 2021-02-23
                warnings.warn(
                    "Use `internal_energies` instead of `slopes`", DeprecationWarning
                )
                self.internal_energies = kwargs.pop("slopes")
            else:
                raise ValueError("Cannot set both `internal_energies` and `slopes`")

        if variables is None:
            if self.dim == 1:
                self.variables = ["phi"]
            else:
                self.variables = [f"phi{i}" for i in range(1, self.num_comp)]
        elif len(variables) == self.num_comp:
            self.variables = list(variables[:-1])
        elif len(variables) == self.dim:
            self.variables = list(variables)
        else:
            raise ValueError(f"`variables` must be a list of {self.dim} strings")

        # deal with variable bounds
        self._set_variable_bounds({"*": self.variable_bounds_default})

        if kwargs:
            raise ValueError(f"Did not use arguments {kwargs}")

    @property
    def chis(self) -> np.ndarray:
        r""" Flory interaction parameters :math:`\chi_{ij}` """
        return self._chis

    @chis.setter
    def chis(self, value: NumberOrArray):
        """ set the interaction parameters """
        shape = (self.num_comp, self.num_comp)
        if np.isscalar(value):
            # a scalar value sets all off-diagonal entries
            chis = np.full(shape, value)

        else:
            chis = np.array(np.broadcast_to(value, shape))
            if not np.allclose(np.diag(chis), 0):
                self._logger.warning("Diagonal part of the chi matrix is not used")

        # ensure that the diagonal entries vanish
        np.fill_diagonal(chis, 0)

        # ensure that the chi matrix is symmetric
        if not np.allclose(chis, chis.T):
            self._logger.warning("Using symmetrized χ interaction-matrix")
        self._chis = 0.5 * (chis + chis.T)

        self._calculate_reduced_values()

    @property
    def internal_energies(self) -> np.ndarray:
        r"""Internal energies :math:`w_i` """
        return self._internal_energies

    @internal_energies.setter
    def internal_energies(self, values: NumberOrArray):
        """ sets the internal energies of the free energy """
        self._internal_energies = np.broadcast_to(values, (self.num_comp,))
        self._calculate_reduced_values()

    def _calculate_reduced_values(self) -> None:
        """ calculate the reduced Flory parameters and internal_energies """
        chis = self.chis
        w = self.internal_energies
        self._chis_reduced = np.empty((self.dim, self.dim))
        self._internal_energies_reduced = np.empty(self.dim)

        n = self.num_comp - 1  # index of the component to be removed
        for i in range(self.dim):
            for j in range(self.dim):
                self._chis_reduced[i, j] = chis[i, j] - chis[i, n] - chis[n, j]
            self._internal_energies_reduced[i] = w[i] - w[n] + chis[i, n]

    def _repr_data(self, formatter=str) -> Tuple[str, Dict[str, Any]]:
        """ return data useful for representing this class """
        data = {
            "num_comp": self.num_comp,
            "chis": formatter(self.chis),
            "sizes": formatter(self.sizes),
            "internal_energies": formatter(self.internal_energies),
        }
        template = (
            "{class}(num_comp={num_comp}, chis={chis}, sizes={sizes} "
            "internal_energies={internal_energies})"
        )
        return template, data

    def __call__(self, *phis, t: float = 0):
        assert len(phis) == self.dim, f"Require {self.dim} fields"
        return self.free_energy(np.array(phis), t)

    @property
    def expression(self) -> str:
        """ str: the mathematical expression describing the free energy """
        # gather all the variables
        var_last = f"(1 - {' - '.join(self.variables)})"
        variables = self.variables + [var_last]

        result = []
        # entropic terms
        for i, (var, size) in enumerate(zip(variables, self.sizes)):
            log_var = f"log({var[1:-1]})" if i == self.dim else f"log({var})"
            if size == 1:
                result.append(f"{var} * {log_var}")
            else:
                result.append(f"{var}/{size:g} * {log_var}")

        # quadratic enthalpic terms
        for i, vi in enumerate(variables):
            for j, vj in enumerate(variables[i:], i):
                if self.chis[i, j] != 0:
                    term = f"{self.chis[i, j]:g} * {vi} * {vj}"
                    result.append(term)

        # linear enthalpic terms
        for i, vi in enumerate(variables):
            if self.internal_energies[i] != 0:
                result.append(f"{self.internal_energies[i]:g} * {vi}")

        return " + ".join(result)

    def free_energy(self, phi: NumberOrArray, t: float = 0) -> NumberOrArray:
        """evaluate the local free energy density

        Args:
            phi: volume fraction at which the free energy is evaluated
            t: simulation time at which the free energy is evaluated

        Returns:
            the free energy associated with `phi`
        """
        phi = np.asanyarray(phi)
        phi_last: np.ndarray = 1 - phi.sum(axis=0)

        return (  # type: ignore
            np.einsum("i,i...->...", 1 / self.sizes[:-1], xlogx(phi))
            + xlogx(phi_last) / self.sizes[self.dim]
            + np.einsum("i...,ij,j...->...", phi, 0.5 * self._chis_reduced, phi)
            + np.einsum("i,i...->...", self._internal_energies_reduced, phi)
            + self.internal_energies[self.dim]
        )

    def chemical_potential(
        self, phi: NumberOrArray, t: float = 0, *, out: np.ndarray = None
    ) -> np.ndarray:
        """evaluate the local part of the chemical potential

        Args:
            phi: volume fraction at which the chemical potential is evaluated
            t: time at which the chemical potential is evaluated

        Returns:
            the chemical potential associated with `phi`
        """
        phi = np.atleast_1d(phi)
        self.regularize_state(phi)
        phi_last = 1 - phi.sum(axis=0)

        if out is None:
            out = np.empty_like(phi)

        for i in range(self.dim):
            out[i] = (1 + np.log(phi[i])) / self.sizes[i]
            for j in range(self.dim):
                out[i] += self._chis_reduced[i, j] * phi[j]
            out[i] += self._internal_energies_reduced[i]
        out -= (1 + np.log(phi_last)) / self.sizes[self.dim]

        return out

    def make_chemical_potential(
        self, backend: str = "numba"
    ) -> Callable[[np.ndarray, float, np.ndarray], np.ndarray]:
        """return function evaluating the chemical potential

        Args:
            backend (str):
                Specifies how the functions are created. Accepted values are 'numpy'
                and 'numba'.

        Returns:
            A function that evaluates the chemical potential
        """
        if backend == "numpy":
            # use straight-forward numpy version
            mu_local = self.chemical_potential

        elif backend == "numba":
            # numba optimized version
            dim = self.dim
            sizes_inv = 1 / self.sizes
            chi_reduced = self._chis_reduced
            internal_energies_reduced = self._internal_energies_reduced

            @jit
            def mu_local(arr: np.ndarray, t: float, out: np.ndarray) -> np.ndarray:
                for index in nb.prange(arr[0].size):
                    # determine solvent component
                    phi_last = 1
                    for i in range(dim):
                        phi_last -= arr[i].flat[index]
                    entropy_last = (1 + np.log(phi_last)) * sizes_inv[dim]

                    for i in range(dim):
                        # calculate chemical potential for species i
                        mu = (1 + np.log(arr[i].flat[index])) * sizes_inv[i]
                        for j in range(dim):
                            mu += chi_reduced[i, j] * arr[j].flat[index]

                        mu += internal_energies_reduced[i] - entropy_last
                        out[i].flat[index] = mu
                return out

        else:
            raise ValueError(f"Backend `{backend}` is not supported")

        return mu_local  # type: ignore

    def make_chemical_potential_split(
        self, backend: str = "numba"
    ) -> Tuple[
        Callable[..., np.ndarray], Callable[..., np.ndarray], Callable[..., np.ndarray]
    ]:
        """return function evaluating the split chemical potential

        This function is useful to implement algorithms based on energy
        splitting where the free energy is split in a convex and a concave part.
        This method returns the associated chemical potentials and their
        derivatives (curvatures of the free energy density).

        Args:
            backend (str):
                Specifies how the functions are created. Accepted values are 'numpy' and
                'numba'.

        Returns:
            tuple: Three functions that evaluates the chemical potentials
        """
        if backend == "numpy":
            # numpy optimized version
            def mu_local_ex(
                phi: np.ndarray, t: float = 0, out: np.ndarray = None
            ) -> np.ndarray:
                phi = np.atleast_1d(phi)
                assert phi.shape[0] == self.dim, f"Require {self.dim} fields"

                if out is None:
                    out = np.empty_like(phi)

                for i in range(self.dim):
                    out[i] = self._internal_energies_reduced[i]
                    for j in range(self.dim):
                        out[i] += self._chis_reduced[i, j] * phi[j]
                return out

            def mu_local_im(
                phi: np.ndarray, t: float = 0, out: np.ndarray = None
            ) -> np.ndarray:
                phi = np.atleast_1d(phi)

                if out is None:
                    out = np.empty_like(phi)

                for i in range(self.dim):
                    out[i] = (1 + np.log(phi[i])) / self.sizes[i]
                out -= (1 + np.log(1 - phi.sum(axis=0))) / self.sizes[self.dim]
                return out

            def mu_local_im_diff(
                phi: np.ndarray, t: float = 0, out: np.ndarray = None
            ) -> np.ndarray:
                phi = np.atleast_1d(phi)

                if out is None:
                    out = np.empty((self.dim,) + phi.shape)

                out[:] = 1 / (1 - phi.sum(axis=0)) / self.sizes[self.dim]
                for i in range(self.dim):
                    out[i, i] += 1 / (phi[i] * self.sizes[i])
                return out

        elif backend == "numba":
            # numba optimized version
            dim = self.dim
            sizes_inv = 1 / self.sizes
            chi_reduced = self._chis_reduced
            internal_energies_reduced = self._internal_energies_reduced

            @jit
            def mu_local_ex(
                arr: np.ndarray, t: float = 0, out: np.ndarray = None
            ) -> np.ndarray:
                if out is None:
                    out = np.empty_like(arr)
                for i in range(dim):
                    # calculate chemical potential for species i
                    mu = internal_energies_reduced[i]
                    for j in range(dim):
                        mu += chi_reduced[i, j] * arr[j]

                    out[i] = mu
                return out

            @jit
            def mu_local_im(
                arr: np.ndarray, t: float = 0, out: np.ndarray = None
            ) -> np.ndarray:
                if out is None:
                    out = np.empty_like(arr)
                # determine solvent component
                entropy_last = (1 + np.log(1 - arr.sum())) * sizes_inv[dim]

                for i in range(dim):
                    # calculate chemical potential for species i
                    mu_i = (1 + np.log(arr[i])) * sizes_inv[i]
                    out[i] = mu_i - entropy_last
                return out

            @jit
            def mu_local_im_diff(
                arr: np.ndarray, t: float = 0, out: np.ndarray = None
            ) -> np.ndarray:
                if out is None:
                    out = np.empty((arr.shape[0],) + arr.shape)
                # determine solvent component
                out[:] = sizes_inv[dim] / (1 - arr.sum())

                for i in range(dim):
                    out[i, i] += sizes_inv[i] / arr[i]
                return out

        else:
            raise ValueError(f"Backend `{backend}` is not supported")

        return mu_local_ex, mu_local_im, mu_local_im_diff
