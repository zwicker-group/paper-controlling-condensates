"""
Module defining base classes for defining free energies 
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>    
"""

import itertools
import logging
import warnings
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numba as nb
import numpy as np
from numba.extending import register_jitable
from scipy import optimize

from pde.fields.base import FieldBase
from pde.tools.cache import cached_method
from pde.tools.plotting import PlotReference, plot_on_axes
from pde.tools.typing import NumberOrArray

TNumArr = TypeVar("TNumArr", float, np.ndarray)


class FreeEnergyBase(metaclass=ABCMeta):
    """ abstract base class for free energies """

    dim: int
    """ int: The number of independent components. For an incompressible system,
    this is typically one less than the number of components. """

    variables: List[str]
    """ list: the names of the variables defining this free energy. The order in this
    list defines the order in which values are supplied to methods of this class"""

    variable_bounds: Dict[str, Tuple[float, float]]
    """ dict: the bounds imposed on each variable """

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def chemical_potential(self, phi: NumberOrArray, t: float = 0) -> np.ndarray:
        pass

    @abstractproperty
    def expression(self) -> str:
        pass

    @abstractmethod
    def __call__(self, *args, t: float = 0):
        pass

    @abstractmethod
    def free_energy(self, phi: NumberOrArray, t: float = 0) -> NumberOrArray:
        pass

    @abstractmethod
    def _repr_data(self) -> Tuple[str, Dict[str, Any]]:
        pass

    @abstractmethod
    def make_chemical_potential(
        self, backend: str = "numba"
    ) -> Callable[[np.ndarray, float, np.ndarray], np.ndarray]:
        pass

    def make_chemical_potential_split(
        self, backend: str = "numba"
    ) -> Tuple[
        Callable[..., np.ndarray], Callable[..., np.ndarray], Callable[..., np.ndarray]
    ]:
        raise NotImplementedError

    def __repr__(self):
        template, data = self._repr_data(formatter=repr)
        data["class"] = self.__class__.__name__
        return template.format(**data)

    def __str__(self):
        template, data = self._repr_data(formatter=str)
        data["class"] = self.__class__.__name__
        return template.format(**data)

    def _concentration_samples(self, num: int) -> np.ndarray:
        """return an array of (uniform) samples of valid concentrations

        Args:
            num (int): The number of samples per dimension

        Returns:
            numpy.ndarray: An array of concentrations. Returns num**dim (or
                less) items of length `dim`, where `dim` is the number of
                independent components.
        """
        c_single = np.linspace(-0.1, 1.1, num)
        cs = np.array(list(itertools.product(c_single, repeat=self.dim)))
        self.regularize_state(cs)
        return np.squeeze(np.unique(cs, axis=0))  # type: ignore

    def regularize_state(self, phi: np.ndarray) -> float:
        """regularize a state ensuring that variables stay within bounds

        Args:
            state (:class:`~numpy.ndarray`):
                The state given as an array of local concentrations

        Returns:
            float: a measure for the corrections applied to the state
        """
        # determine the bounds for the variable
        bounds = self.variable_bounds[self.variables[0]]

        if np.all(np.isinf(bounds)):
            # there are no bounds to enforce
            return 0

        # check whether the state is finite everywhere
        if not np.all(np.isfinite(phi)):
            raise RuntimeError("State is not finite")

        # ensure all variables are positive are in (0, 1)
        np.clip(phi, *bounds, out=phi)

        # TODO: Return the correct amount of regularization applied
        return np.nan

    def make_state_regularizer(
        self, state: FieldBase, global_adjust: bool = False
    ) -> Callable[[np.ndarray], float]:
        """returns a function that can be called to regularize a state

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            global_adjust (bool):
                Flag indicating whether we attempt to preserve the total amount of
                material by adjusting the fields globally.

        Returns:
            Function that can be applied to a state to regularize it and which
            returns a measure for the corrections applied to the state
        """
        if all(np.isinf(a) and np.isinf(b) for a, b in self.variable_bounds.values()):
            self._logger.info("Skip regularizer since no bounds are present")

            # no bounds need to be enforced
            def regularizer_noop(phi: np.ndarray) -> float:
                """ no-op regularizer """
                return 0

            return regularizer_noop

        # we need to enforce bounds for multiple variables
        dim = self.dim
        bounds = np.array([self.variable_bounds[v] for v in self.variables])
        assert bounds.shape == (dim, 2)
        self._logger.info("Use regularizer enforcing bounds %s", self.variable_bounds)

        vol_sys = state.grid.volume
        vol = state.grid.make_cell_volume_compiled(flat_index=True)

        @register_jitable
        def regularizer_inner(phi: np.ndarray, phi_min: float, phi_max: float) -> float:
            """ helper function ensuring a single species stays in a given bound """
            # accumulate lower and upper bound separately; both values are positive
            amount_low = 0
            vol_low = 0
            amount_high = 0
            vol_high = 0

            # determine the deviation amount
            for i in range(phi.size):
                if phi.flat[i] < phi_min:
                    # concentration is below lower bound
                    v = vol(i)
                    amount_low += v * (phi_min - phi.flat[i])  # type: ignore
                    vol_low += v
                    phi.flat[i] = phi_min  # type: ignore

                elif phi.flat[i] > phi_max:
                    # concentration is above upper bound
                    v = vol(i)
                    amount_high += v * (phi.flat[i] - phi_max)  # type: ignore
                    vol_high += v
                    phi.flat[i] = phi_max  # type: ignore

            # correct the data if requested
            if global_adjust:
                if amount_high > amount_low:
                    # we had more points that were too high => add material elsewhere
                    conc_corr = (amount_high - amount_low) / (vol_sys - vol_high)
                    assert conc_corr > 0
                    for i in range(phi.size):
                        phi.flat[i] = min(phi_max, phi.flat[i] + conc_corr)  # type: ignore

                elif amount_high < amount_low:
                    # we had more points that were too low => remove material elsewhere
                    conc_corr = (amount_low - amount_high) / (vol_sys - vol_low)
                    assert conc_corr > 0
                    for i in range(phi.size):
                        phi.flat[i] = max(phi_min, phi.flat[i] - conc_corr)  # type: ignore

                # else:
                #     both amounts are equal and cancel each other

            # return the total amount that was corrected anywhere
            return amount_high + amount_low

        if self.dim == 1:
            # a single species => array is not nested
            phi_min, phi_max = bounds[0]

            def regularizer(phi: np.ndarray) -> float:
                """ ensure all variables are positive are in (0, 1) """
                return regularizer_inner(phi, phi_min, phi_max)  # type: ignore

        else:
            # multiple species => correct each individual species
            def regularizer(phi: np.ndarray) -> float:
                """ ensure all variables are positive are in (0, 1) """
                # iterate over all species
                correction = 0
                for j in nb.prange(dim):
                    phi_min, phi_max = bounds[j]
                    correction += regularizer_inner(phi[j], phi_min, phi_max)
                return correction

        return regularizer

    def curvature(self, phi: NumberOrArray, t: float = 0) -> NumberOrArray:
        """calculate the curvature of the free energy at the given point

        Args:
            phi (:class:`~numpy.ndarray`):
                The volume fraction / concentration at which the curvature is
                calculated

        Returns:
            The curvature of the local free energy density
        """
        if self.dim != 1:
            raise NotImplementedError(
                "Curvature cannot be calculated for "
                f"free energies with {self.dim} "
                "independent variables."
            )
        eps = 1e-8
        mu0 = self.chemical_potential(phi - eps, t=t)
        mu1 = self.chemical_potential(phi + eps, t=t)
        return (mu1 - mu0) / (2 * eps)  # type: ignore

    def get_spinodal(self, num_points: int = 5, t: float = 0) -> List[float]:
        """return the concentrations of the spinodal line

        This is looking for solutions `c` that fulfill :math:`f''(c) == 0`

        Args:
            num_points (int): The number of initial seeds along the interval
                [0, 1], which are used in the numerical root finding.
            t (float):
                The current time point

        Returns:
            A list of the distinct solutions that were found
        """
        if self.dim != 1:
            raise NotImplementedError(
                "Spinodals only implemented for one-" "component free energy densities"
            )

        result: List[float] = []
        for c in self._concentration_samples(num_points):
            # solve the equilibrium conditions
            sol = optimize.root(lambda c: self.curvature(c, t=t), c)
            if not sol.success or np.abs(sol.fun) > 1e-8:
                self._logger.debug("Getting spinodal failed: %s", sol.message)
                continue
            c_sol = np.array(sol.x, copy=True)

            self._logger.debug("Spinodal candidate: %s", c_sol)

            # skip solutions close to those previously found
            if any(np.isclose(v, c_sol[0]) for v in result):
                continue

            # check whether regularizing the solutions changes them
            self.regularize_state(c_sol)
            if not np.isclose(sol.x, c_sol):
                continue

            result.append(c_sol[0])

        return result

    @cached_method()
    def get_equilibrium(
        self, laplace_pressure: float = 0, num_points: int = 5, t: float = 0
    ) -> List[Tuple[float, float]]:
        r""" determine the equilibrium conditions for the current free energy.
        
        The equilibrium solutions :math:`(c_1, c_2)` fulfill
        
        .. math ::
            0 &= f'(c_1) - f'(c_2) \\
            0 &= f(c_2) - f(c_1) + (c_1 - c_2)  f'(c_1) + \Pi
            
        where :math:`\Pi` is the Laplace pressure. For :math:`\Pi=0`, the
        solutions correspond to the binodal line.
        
        Args:
            laplace_pressure(float): Laplace pressure applied to the interface
                during equilibration
            num_points (int): The number of initial seeds along the interval
                [0, 1], which are used in the numerical minimization. This value
                is only used if local minima of the free energy could not be
                determined.
            t (float):
                The current time point
            
        Returns:
            a list of solutions `(c_1, c_2)` for the concentrations in the two
            phases. Multiple results should only be returned when the underlying
            free energy density has more than two minima.
        """
        if self.dim != 1:
            raise NotImplementedError(
                "Equilibration only implemented for one-"
                "component free energy densities"
            )

        # generate candidates for concentrations close to the equilibrium by
        # first trying to locate local minima in a tilted version of the free
        # energy density. If this did not work, we simply use a few values
        # uniformly spaced values

        # determine average slope of f(c)
        x = self._concentration_samples(16)
        y = self(x)
        slope_est = np.polyfit(x, y, deg=1)[0]

        def mu_corr(c):
            """ derivative of tilted free energy density """
            val = self.chemical_potential(c, t=t) - slope_est
            deriv = self.curvature(c, t=t)
            return val, deriv

        # determine the minimal values of f with corrected slope
        c_samples = self._concentration_samples(num_points)
        cs = []
        for c in c_samples:
            res = optimize.root_scalar(mu_corr, x0=c, fprime=True)
            if res.converged:
                cs.append(res.root)

        # make sure there are enough candidate points
        if len(cs) < 2:
            cs_arr = c_samples
        else:
            cs_arr = np.unique(cs)

        def eqs(cs):
            """ private function defining the equilibrium conditions """
            self.regularize_state(cs)
            c1, c2 = cs
            mu1 = self.chemical_potential(c1, t=t)
            mu2 = self.chemical_potential(c2, t=t)
            mu = (mu1 + mu2) / 2

            eq1 = mu1 - mu2
            eq2 = self(c2) - self(c1) + (c1 - c2) * mu + laplace_pressure

            jac11 = self.curvature(c1, t=t)
            jac12 = -self.curvature(c2, t=t)
            jac21 = -1 + mu + (c1 - c2) / 2 * self.curvature(c1, t=t)
            jac22 = 1 - mu + (c1 - c2) / 2 * self.curvature(c2, t=t)
            return [eq1, eq2], [[jac11, jac12], [jac21, jac22]]

        # determine initial conditions for root finding
        cs1, cs2 = np.meshgrid(cs_arr, cs_arr)
        cs1, cs2 = cs1.ravel(), cs2.ravel()
        i = cs1 < cs2
        cs1, cs2 = cs1[i], cs2[i]

        result: List[Tuple[float, float]] = []
        for c1, c2 in zip(cs1, cs2):
            # solve the equilibrium conditions
            sol = optimize.root(eqs, (c1, c2), jac=True)
            if not sol.success:
                self._logger.debug("Equilibration failed: %s", sol.message)
                continue
            cs = sol.x

            # sort solutions
            if cs[0] > cs[1]:
                cs = cs[1], cs[0]  # type: ignore

            self._logger.debug("Equilibrium candidate: %s", cs)

            # skip solutions where concentrations identical
            if np.isclose(cs[0], cs[1], rtol=1e-3, atol=1e-3):
                continue
            # skip solutions where the free energy is convex
            if any(self.curvature(c, t=t) < 0 for c in cs):  # type: ignore
                continue
            # skip solutions close to those previously found
            if any(np.allclose(cs, v) for v in result):
                continue
            result.append(cs)  # type: ignore

        return result

    @plot_on_axes()
    def plot(
        self,
        ax,
        extent: Tuple[float, float] = (1e-6, 1 - 1e-6),
        laplace_pressure: Optional[float] = None,
        t: float = 0,
        **kwargs,
    ) -> PlotReference:
        """visualizes the free energy density

        If `laplace_pressure` is given, the Maxwell construction is shown, too.

        Args:
            extent (tuple):
                The coordinate range that is shown
            laplace_pressure (float, optional):
                Laplace pressure applied to the interface during equilibration.
                The default value `None` implies that the Maxwell construction
                is not shown. Setting this to zero, shows the Maxwell
                construction for flat interface, whereas positive values also
                consider surface tension effects.
            t (float):
                The current time point
            {PLOT_ARGS}

        Returns:
            :class:`pde.tools.plotting.PlotReference`: Instance that contains
            information about the plot.
        """
        if self.dim == 1:
            # plot free energy depending on one concentration
            cs = np.linspace(*extent, 128)
            kwargs.setdefault("color", "k")
            (line,) = ax.plot(cs, self(cs, t=t), **kwargs)

            ax.set_xlabel(self.variables[0])
            ax.set_ylabel("Free energy density")

            if laplace_pressure is not None:
                if self.dim != 1:
                    raise NotImplementedError(
                        "Maxwell plot only implemented for "
                        "one-component free energy densities"
                    )

                ceqs = self.get_equilibrium(laplace_pressure=laplace_pressure, t=t)
                if len(ceqs) == 0:
                    self._logger.warn(
                        "Did not find any equilibrium solutions for "
                        f"Laplace pressure of {laplace_pressure}"
                    )
                for ceq in ceqs:
                    for c in ceq:
                        y = self(c, t=t) + self.chemical_potential(c, t=t) * (cs - c)
                        ax.plot(cs, y, color="C1")

            return PlotReference(ax, line)

        elif self.dim == 2:
            # plot free energy depending on two concentrations

            colorbar = kwargs.get("colorbar", True)
            kwargs.setdefault("origin", "lower")

            cs = np.linspace(*extent, 128)
            xs, ys = np.meshgrid(cs, cs)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f = self(xs, ys, t=t)

            img = ax.imshow(f, extent=extent * 2, **kwargs)

            ax.set_xlabel(self.variables[0])
            ax.set_ylabel(self.variables[1])

            if colorbar:
                from pde.tools.plotting import add_scaled_colorbar

                add_scaled_colorbar(img, ax=ax)

            return PlotReference(ax, img)

        else:
            raise NotImplementedError(
                f"Cannot plot free energy depending on {self.dim} components"
            )

    @plot_on_axes()
    def plot_ternary(
        self,
        ax,
        t: float = 0,
        variable_3: str = "Solvent",
        **kwargs,
    ) -> PlotReference:
        """visualizes the free energy density

        Args:
            t (float):
                The current time point
            variable_3 (str):
                The name of the third variable
            {PLOT_ARGS}

        Returns:
            :class:`pde.tools.plotting.PlotReference`: Instance that contains
            information about the plot.
        """
        if self.dim == 2:
            # plot free energy depending on two concentrations
            try:
                import ternary
            except ImportError:
                self._logger.exception(
                    "The `ternary` python package is not available. Please install it "
                    "using `pip install python-ternary` or a similar command."
                )

            scale = 10
            tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)

            def calc_f(p):
                """ helper function calculating the free energy """
                try:
                    return self(p[0], p[1], t=t)
                except FloatingPointError:
                    return np.nan

            heatmap = tax.heatmapf(
                calc_f, scale=scale, boundary=True, cbarlabel="Free energy", **kwargs
            )
            tax.ticks([f"{v:g}" for v in np.linspace(0, 1, scale + 1)])
            tax.boundary(linewidth=2.0)
            tax.get_axes().axis("off")

            tax.left_axis_label(self.variables[0])
            tax.right_axis_label(self.variables[1])
            tax.bottom_axis_label(variable_3)

            return PlotReference(ax, heatmap)

        else:
            raise NotImplementedError(
                f"Cannot plot ternary diagram depending on {self.dim} components"
            )


class FreeEnergyNComponentsBase(FreeEnergyBase, metaclass=ABCMeta):
    """ abstract base class for free energies of multiple components """

    def _set_variable_bounds(self, variable_bounds):
        # deal with variable bounds
        self.variable_bounds = {}
        default_bound = variable_bounds.get("*", None)
        for variable in self.variables:
            bound = variable_bounds.get(variable, default_bound)
            if bound is None:
                raise RuntimeError(f"Could not determine bound for `{variable}`")
            self.variable_bounds[variable] = bound

    def regularize_state(self, phi: np.ndarray, sum_max: float = 1 - 1e-8) -> float:
        """regularize a state ensuring that variables stay within bounds

        The bounds for all variables are defined in the class attribute
        :attr:`variable_bounds`.

        Args:
            phi (:class:`~numpy.ndarray`):
                The state given as an array of local concentrations
            sum_max (float):
                The maximal value the sum of all concentrations may have. This can be
                used to limit the concentration of a variable that has been removed due
                to incompressibility. If this value is set to `np.inf`, the constraint
                is not applied

        Returns:
            float: a measure for the corrections applied to the state
        """
        if not np.all(np.isfinite(phi)):
            raise RuntimeError("State is not finite")

        if self.dim == 1:
            # deal with a single variable
            return super().regularize_state(phi)

        else:
            # deal with multiple variables

            # adjust each variable individually
            for i, variable in enumerate(self.variables):
                bounds = self.variable_bounds[variable]
                phi[i] = np.clip(phi[i], *bounds)
                # Note that we did not use the `out` argument, since this would not work
                # if `phi[i]` was a scalar

            # limit the sum of all variables
            if np.isfinite(sum_max):
                phis = phi.sum(axis=0)
                loc = phis > sum_max
                if np.any(loc):
                    phi[:, loc] *= sum_max / phis[loc]

        # TODO: Return the correct amount of regularization applied
        return np.nan

    def make_state_regularizer(
        self, state: FieldBase, sum_max: float = 1 - 1e-8
    ) -> Callable[[np.ndarray], float]:
        """returns a function that can be called to regularize a state

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other
                information can be extracted
            sum_max (float):
                The maximal value the sum of all concentrations may have. This can be
                used to limit the concentration of a variable that has been removed due
                to incompressibility. If this value is set to `np.inf`, the constraint
                is not applied

        Returns:
            Function that can be applied to a state to regularize it and which
            returns a measure for the corrections applied to the state
        """
        if self.dim == 1:
            # deal with a single variable
            return super().make_state_regularizer(state)

        else:
            # deal with multiple variables
            dim = self.dim
            bounds = np.array(
                [self.variable_bounds[variable] for variable in self.variables]
            )

            def regularizer(phi: np.ndarray) -> Callable[[np.ndarray], float]:
                """ regularize a state ensuring variables stay within bounds """
                if not isinstance(phi, (np.ndarray, nb.types.Array)):
                    raise TypeError

                if phi.ndim == 1:
                    # a single set of concentrations is given

                    def regularizer_impl(phi: np.ndarray) -> float:
                        """ regularize a state ensuring variables stay within bounds """
                        correction = 0.0

                        # adjust each variable individually
                        for i in range(dim):
                            if phi[i] < bounds[i, 0]:
                                correction += bounds[i, 0] - phi[i]
                                phi[i] = bounds[i, 0]
                            elif phi[i] > bounds[i, 1]:
                                correction += phi[i] - bounds[i, 1]
                                phi[i] = bounds[i, 1]

                        # limit the sum of all variables
                        if np.isfinite(sum_max):
                            phis = 0.0
                            for i in range(dim):
                                phis += phi[i]
                            if phis > sum_max:
                                for i in range(dim):
                                    phi[i] *= sum_max / phis

                        return correction

                else:
                    # an array of concentrations is given

                    def regularizer_impl(phi: np.ndarray) -> float:
                        """ regularize a state ensuring variables stay within bounds """
                        correction = 0.0

                        # adjust each variable individually
                        for i in range(dim):
                            for j in range(phi[0].size):
                                if phi[i].flat[j] < bounds[i, 0]:
                                    correction += bounds[i, 0] - phi[i].flat[j]
                                    phi[i].flat[j] = bounds[i, 0]
                                elif phi[i].flat[j] > bounds[i, 1]:
                                    correction += phi[i].flat[j] - bounds[i, 1]
                                    phi[i].flat[j] = bounds[i, 1]

                        # limit the sum of all variables
                        if np.isfinite(sum_max):
                            for j in range(phi[0].size):
                                phis = 0.0
                                for i in range(dim):
                                    phis += phi[i].flat[j]
                                if phis > sum_max:
                                    for i in range(dim):
                                        phi[i].flat[j] *= sum_max / phis

                        return correction

                return regularizer_impl

            if nb.config.DISABLE_JIT:
                # jitting is disabled => return generic python function

                # we here simply supply a 2d array so the more generic implementation
                # is chosen, which works for all cases in the case of numpy
                return regularizer(np.empty((2, 2)))

            else:
                # jitting is enabled => return specialized, compiled function
                return nb.generated_jit(nopython=True)(regularizer)  # type: ignore


def get_free_energy_single(
    free_energy: Union[str, FreeEnergyBase] = "ginzburg-landau"
) -> FreeEnergyBase:
    """get free energy for systems with a single effective component

    Args:
        free_energy (str or FreeEnergyBase):
            Defines the expression to used for the local part of the free energy
            density. This can either be string for common choices of free
            energies ('ginzburg-landau' or 'flory-huggins'), or an instance of
            :class:`~phasesep.free_energies.base.FreeEnergyBase`,
            which provides methods for evaluating the local free energy density
            and chemical potentials.

    Returns:
        FreeEnergyBase: An instance of
        :class:`~phasesep.free_energies.base.FreeEnergyBase` that
        represents the free energy
    """
    if free_energy == "ginzburg-landau":
        from .ginzburg_landau import GinzburgLandau2Components

        f_local: FreeEnergyBase = GinzburgLandau2Components()
    elif free_energy == "flory-huggins":
        from .flory_huggins import FloryHuggins2Components

        f_local = FloryHuggins2Components()
    elif isinstance(free_energy, str):
        raise ValueError(f"Free energy `{free_energy}` is not defined")
    else:
        f_local = free_energy

    # check some properties for consistency
    if f_local.dim != 1:
        raise ValueError(f"Too many components ({f_local.dim})")
    from .general import FreeEnergy

    if isinstance(f_local, FreeEnergy) and not f_local.squeeze_dims:
        raise ValueError(
            "Free energy for single component must have `squeeze_dims=True`."
        )
    return f_local
