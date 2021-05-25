"""
Defines a class representing the
`Cahn-Hilliard equation <https://en.wikipedia.org/wiki/Cahn–Hilliard_equation>`_
for multiple species.
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from numba.core.errors import NumbaExperimentalFeatureWarning, NumbaPerformanceWarning

from pde.fields import FieldCollection, ScalarField, VectorField
from pde.fields.base import FieldBase
from pde.grids.boundaries.axes import BoundariesData
from pde.grids.cartesian import CartesianGridBase
from pde.pdes import PDEBase
from pde.tools.docstrings import get_text_block
from pde.tools.numba import jit, nb
from pde.tools.parameters import ParameterListType  # @UnusedImport
from pde.tools.parameters import Parameter, Parameterized

from .reactions import Reaction, Reactions

BoundariesDataList = Union[BoundariesData, Sequence[BoundariesData]]


def _matrix_decomposition(arr: np.ndarray) -> np.ndarray:
    """turns a matrix A into another matrix B, such that B @ B.T == A

    Args:
        arr (:class:`~numpy.array`): The input matrix

    Results:
        :class:`~numpy.array`: The output
    """
    if np.allclose(arr, np.diag(np.diagonal(arr))):
        return np.sqrt(arr)  # type: ignore
    else:
        return np.linalg.cholesky(arr)  # type: ignore


class CahnHilliardMultiplePDE(PDEBase, Parameterized):
    r"""(extended) incompressible Cahn-Hilliard equation for many components

    The extended Cahn-Hilliard equation describes the temporal evolution of
    :math:`N` components, described by their volume fraction fields,
    :math:`\boldsymbol\phi(\boldsymbol r, t) = \{\phi_0, \phi_1, \ldots,
    \phi_{N-1}\}`.
    The dynamics are given by

    .. math::
        \partial_t \phi_i = \partial_{\alpha}\biggl(
                    \sum_{j=0}^{N-1} \Lambda_{ij}(\boldsymbol \phi)
                        \partial_{\alpha}\mu_j
                \biggr) +
                s_i(\boldsymbol\phi, \boldsymbol\mu, t)

    Here, :math:`\Lambda_{ij}(\boldsymbol\phi)` is the mobility matrix, which
    can depend on the local composition vector :math:`\boldsymbol\phi`,
    :math:`s_i(\boldsymbol\phi, \boldsymbol\mu, t)` gives the
    reaction rate, and the chemical potentials :math:`\mu_i` are given by the
    functional derivatives
    :math:`\mu_i = \delta F[\boldsymbol\phi]/\delta \phi_i` of the free energy

    .. math::
        F[\boldsymbol\phi] = \int \biggl(
                f(\boldsymbol\phi) +
                \sum_{i,j=0}^{N-1}\frac{\kappa_{ij}}{2}
                    (\partial_\alpha \phi_i)(\partial_\alpha \phi_j)
            \biggr) \mathrm{d}V

    where :math:`f(\boldsymbol\phi)` is the free energy density of the :math:`N`
    components, e.g., given by
    :class:`~phasesep.free_energies.flory_huggins.
    FloryHugginsNComponents`, and :math:`\kappa_{ij}` is a matrix describing how
    gradients in the concentration profiles affect the free energy.

    Note, that we assume an incompressible system, implying that the component
    :math:`\phi_{N-1} = 1 - \sum_{i=0}^{N-2} \phi_i` can be eliminate and only
    the fields :math:`\phi_i` for :math:`i=0, \ldots, N-2` are specified.
    Consequently, the chemical potentials used in the description are exchange
    chemical potentials :math:`\bar\mu_i = \mu_i - \mu_{N-1}` describing the
    difference to the chemical potential of the removed component.
    """

    parameters_default: ParameterListType = [
        Parameter(
            "free_energy",
            None,
            object,
            "Defines the expression for the local part of the free energy density. "
            "Currently, this needs to be an instance of "
            ":class:`~phasesep.free_energies.flory_huggins.FloryHugginsNComponents`.",
        ),
        Parameter(
            "mobility",
            np.array(1.0),
            np.array,
            "These mobilities define how fast the field relaxes by diffusive fluxes. "
            "In principle, the mobilities should be a matrix, where the off-diagonal "
            "elements correspond to cross-diffusion. If only a 1d-array is specified, "
            "it is assumed that this array defines the diagonal elements while the "
            "off-diagonal elements are zero. Generally, the interpretation of "
            "`mobility` also depends on the `mobility_model` parameter. For instance, "
            "if a single value for `mobility` is given, it is used for all diagonal "
            "elements if the mobility model is 'constant' and as a full matrix if the "
            "mobility model is 'kramer'.",
        ),
        Parameter(
            "kappa",
            1.0,
            object,
            "Pre-factor `kappa_{ij}` for the gradient term in the free energy, which "
            "gives rise to surface tension effects. The actually value used in the "
            "simulation also depends on the parameter `kappa_from_chis` described "
            "below. Generally, `kappa` can be a single number (all interactions have "
            "the same prefactor) or a (symmetric) matrix of dimensions `num_comp` "
            "(specifying different interactions for each gradient combination).",
        ),
        Parameter(
            "reactions",
            None,
            object,
            r"Defines the reaction rates :math:`s_i(\phi, \mu, t)` describing how the "
            "concentration of each component :math:`i` changes locally. Multiple "
            "reactions can be specified by supplying an instance of "
            ":class:`phasesep.reactions.Reactions`. If an instance of "
            ":class:`phasesep.reactions.Reaction` is supplied it is used as the only "
            "reaction in the system. The special value `None` corresponds to no "
            "reactions.",
        ),
        Parameter(
            "bc_phi",
            "natural",
            object,
            "Defines the boundary condition on the concentration (volume fraction) "
            "fields. A list of conditions (one for each field) or a single set of "
            "boundary conditions (the same for each field) can be supplied. "
            + get_text_block("ARG_BOUNDARIES"),
        ),
        Parameter(
            "bc_mu",
            "natural",
            object,
            "Boundary conditions on the chemical potential fields. This supports the "
            "same options as `bc_phi`.",
        ),
        Parameter(
            "mobility_model",
            "constant",
            str,
            "Influences how the `mobility` parameter is interpreted. The only options "
            "are `constant` and `kramer`. The value `constant` implies constant "
            "mobilities directly given by the parameter `mobility`. In contrast, the "
            "value `kramer` chooses a mobility that depends on volume fractions "
            r"according to Kramer's model, :math:`\Lambda_{ij} = D_{ij} ("
            r"\phi_i\delta_{ij} - \phi_i\phi_j)`. Here, :math:`D_{ij}` is the matrix "
            "of diffusivities, which is set by the `mobility` parameter.",
        ),
        Parameter(
            "rhs_implementation",
            "auto",
            str,
            "Determines how the right hand side of the PDE is implemented when the "
            "mobility model is not `constant`. Possible choices are `direct`, where "
            "the flux is calculated explicitly, `split`, where the divergence is "
            "expanded (This does not conserve mass!), or `staggered` which uses a "
            "staggered grid that is only implemented for Cartesian grids and some "
            "boundary conditions. The choice `auto` chooses a suitable value "
            "automatically using the method `determine_rhs_implementation_method`.",
        ),
        Parameter(
            "kappa_from_chis",
            True,
            bool,
            "Determines whether the scaling factors in front of the gradient terms in "
            "the free energy should be determined from the Flory-Huggins interaction "
            r"parameters :math:`\chi_{ij}` of the free energy density. If `True`, the "
            r"actually used factors are :math:`\kappa_{ij} \chi_{ij}`, which allows to "
            r"scale the gradient interactions proportional to :math:`\chi_{ij}` by "
            "specifying a scalar value for `kappa`.",
        ),
        Parameter(
            "implicit_iterations_max",
            1000,
            int,
            "The maximal number of iterations in the implicit step of the energy "
            "splitting method.",
        ),
        Parameter("noise", 0, float, hidden=False),
        Parameter(
            "noise_diffusion",
            np.array([0.0]),
            np.array,
            "Magnitude of the diffusive noise. This is related to the standard "
            "deviation of the Gaussian white noise added to the diffusive flux, which "
            "is also multiplied by the mobility matrix. Consequently, the "
            "thermodynamically consistent value is noise_diffusion=2. However, by "
            "supplying an array, different noise strengths can be imposed for "
            "different fields. In particular, setting the noise strength to zero makes "
            "the respective field not experience any diffusive noise.",
        ),
        Parameter(
            "noise_reaction",
            np.array([0.0]),
            np.array,
            "Magnitude of additive Gaussian noise that models fluctuations in the "
            "reactions.",
        ),
        Parameter(
            "regularize_after_step",
            False,
            bool,
            "Regularize phase field after each time step when the free energy is not "
            "defined for all concentration values. This is a cheap way to enforce "
            "concentration bounds, but it comes at the disadvantage of possibly "
            "violating material conservation",
        ),
    ]

    _kappa: np.ndarray
    _mobilities: np.ndarray

    def __init__(self, parameters: Dict[str, Any] = None):
        r"""
        Args:
            parameters (dict):
                A dictionary of parameters to change the defaults. The allowed
                parameters can be obtained from
                :meth:`~Parameterized.get_parameters` or displayed by calling
                :meth:`~Parameterized.show_parameters`.
        """

        Parameterized.__init__(self, parameters)
        PDEBase.__init__(self, noise=0)

        # the parameter `noise` is ambiguous – do not interpret it
        if self.parameters["noise"] != 0:
            self._logger.warning(
                "Use the parameters `noise_diffusion` and `noise_reaction` to set "
                "noise strength in CahnHilliardExtendedPDE"
            )

        # define the functions for evaluating the right hand side
        if self.parameters["free_energy"] is None:
            raise ValueError(
                "No free energy specified. Define one using the classes `FreeEnergy` "
                "or `FloryHugginsNComponents` and supply it using the `free_energy` "
                "parameter."
            )
        else:
            self.f_local = self.parameters["free_energy"]
        self.dim = self.f_local.dim
        self.num_comp = self.dim + 1

        # set the diffusive and reactive noise strength
        if self.parameters["noise_diffusion"].size == 1:
            self.noise_diffusion = np.full(self.dim, self.parameters["noise_diffusion"])
        elif len(self.parameters["noise_diffusion"]) == self.dim:
            self.noise_diffusion = np.array(self.parameters["noise_diffusion"])
        else:
            raise RuntimeError("Incorrect length of parameter 'noise_diffusion'")

        if self.parameters["noise_reaction"].size == 1:
            self.noise_reaction = np.full(self.dim, self.parameters["noise_reaction"])
        elif len(self.parameters["noise_reaction"]) == self.dim:
            self.noise_reaction = np.array(self.parameters["noise_reaction"])
        else:
            raise RuntimeError("Incorrect length of parameter 'noise_reaction'")

        # get gradient prefactor matrix
        self.kappa = self.parameters["kappa"]

        # set the mobilities
        self.mobilities = self.parameters["mobility"]

        # check the data for the chemical reactions
        if isinstance(self.parameters["reactions"], Reaction):
            reactions = [self.parameters["reactions"]]
        else:
            reactions = self.parameters["reactions"]
        self.reactions = Reactions(self.num_comp, reactions)
        self.explicit_time_dependence = self.reactions.explicit_time_dependence

        # set the boundary conditions
        if isinstance(self.parameters["bc_phi"], (str, dict)):
            self.bc_phi = [self.parameters["bc_phi"]] * self.dim
        else:
            self.bc_phi = self.parameters["bc_phi"]
        if isinstance(self.parameters["bc_phi"], (str, dict)):
            self.bc_mu = [self.parameters["bc_phi"]] * self.dim
        else:
            self.bc_mu = self.parameters["bc_phi"]

    @property
    def is_sde(self) -> bool:
        """flag indicating whether this is a stochastic differential equation

        The :class:`CahnHilliardExtendedPDE` class supports diffusive and
        reactive noise, whose magnitudes are controlled by the `noise_reaction`
        and the `noise_diffusion` property. In this case, `is_sde` is `True` if
        (`self.noise_diffusion != 0` or `self.noise_reaction != 0`).
        """
        # check if general noise parameter was used in class definition
        if self.noise != 0:
            raise RuntimeError(
                "Use the parameters `noise_diffusion` and "
                "`noise_reaction` to set noise in this class."
            )
        # check for self.noise, in case __init__ is not called in a subclass
        return any(self.noise_diffusion) or any(self.noise_reaction)

    @property
    def kappa(self) -> np.ndarray:
        """ numpy.ndarray: Pre-factors for the gradient terms """
        return self._kappa

    @kappa.setter
    def kappa(self, value: Union[float, np.ndarray]):
        """ set the kappa matrix ensuring the correct symmetries """
        value_arr = np.broadcast_to(value, (self.num_comp, self.num_comp))

        if self.parameters["kappa_from_chis"]:
            # scale the given kappa with the chis parameters
            try:
                chis = self.f_local.chis
            except AttributeError:
                self._logger.warning(
                    "Free energy did not define the parameter `chis`. Kappa parameter "
                    "will not be scaled."
                )
            else:
                assert chis.shape == (self.num_comp, self.num_comp)
                value_arr = value_arr * chis

        kappa = 0.5 * (value_arr + value_arr.T)  # symmetrize array
        if np.any(kappa < 0):
            self._logger.warning(
                "Negative kappa parameters promote interfacial instabilities."
            )
        self._kappa = kappa

        # calculate reduced kappa matrix to be used in chemical potential
        self._kappa_reduced = np.empty((self.dim, self.dim))
        n = self.num_comp - 1  # index of the component to be removed
        for i in range(self.dim):
            for j in range(self.dim):
                # the 1/2 corrects the double counting of off-diagonal terms
                self._kappa_reduced[i, j] = (
                    -(kappa[i, j] - kappa[i, n] - kappa[n, j] + kappa[n, n]) / 2
                )

    @property
    def mobilities(self) -> np.ndarray:
        """ numpy.ndarray: The mobilities of the diffusive fluxes """
        return self._mobilities

    @mobilities.setter
    def mobilities(self, mobilities: Union[float, np.ndarray]):
        """sets the mobility matrix

        Args:
            mobilities (float or :class:`~numpy.ndarray`):
                The mobilities are generally given by a matrix, where the off-diagonal
                elements correspond to cross-diffusion. When only a 1d-array is
                specified it is assumed that this array defines the diagonal elements
                while the off-diagonal elements are zero. Finally, if a single value is
                given, it is used for all diagonal elements if the mobility model is
                "constant" or as a full matrix if the mobility model is "kramer".
        """
        # the interpretation of the mobilities depends on the mobility model
        mobility_model = self.parameters["mobility_model"]

        mobilities_arr = np.asarray(mobilities)

        # handle scalar values
        if mobilities_arr.ndim == 0:
            if mobility_model == "constant":
                # interpret value as a diagonal matrix
                mobilities_arr = mobilities_arr * np.eye(self.dim)
            elif mobility_model == "kramer":
                # interpret value as full matrix
                mobilities_arr = np.full((self.dim, self.dim), mobilities_arr)
            else:
                raise ValueError(f"Unknown mobility model `{mobility_model}`")

        # handle multidimensional values
        if mobilities_arr.ndim == 1:
            mobilities_arr = np.diag(mobilities_arr)
        if mobilities_arr.shape != (self.dim, self.dim):
            raise ValueError(
                f"Mobility matrix must be a square matrix of dimension {self.dim}"
            )

        # symmetrize the matrix
        mobilities_unsym = mobilities_arr
        mobilities_arr = 0.5 * (mobilities_arr + mobilities_arr.T)
        if not np.allclose(mobilities_arr, mobilities_unsym):
            self._logger.warning("Mobility matrix was symmetrized")

        # check whether the matrix is consistent with the requirements
        if mobility_model == "constant":
            # matrix must be positive definite for constant mobilities
            is_diag = np.allclose(mobilities_arr, np.diag(np.diagonal(mobilities_arr)))
            if not is_diag and np.any(np.linalg.eigvals(mobilities_arr) <= 0):
                raise ValueError("The mobility matrix must be positive definite")

        elif mobility_model == "kramer":
            # matrix is less restrictive for kramer's model
            if np.any(mobilities_arr < 0):
                raise ValueError("All mobilities must be non-negative")

        else:
            raise ValueError(f"Unknown mobility model `{mobility_model}`")

        self._mobilities = mobilities_arr

    @property
    def info(self) -> Dict[str, Any]:
        """ dict: information about the PDE """
        return self.parameters.copy()

    def _check_field_type(self, state: FieldCollection):
        """checks whether the supplied field is consistent with this class

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                The fields describing the concentration distributions
        """
        if not isinstance(state, FieldCollection):
            raise ValueError("Supplied state must be a FieldCollection")
        if len(state) != self.dim:
            raise ValueError(f"Expected state with {self.dim} fields")

    def phase_fields(self, state: FieldCollection) -> FieldCollection:
        r"""return phase fields for all components including :math:`\phi_{N-1}`

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                The fields describing the concentration distributions

        Returns:
            :class:`~pde.fields.FieldCollection`:
            The concentrations for all `num_comp = dim + 1` fields. The
            additional component is appended, so that `result[:-1] == state`.
        """
        self._check_field_type(state)
        field_data: np.ndarray = 1 - state.data.sum(axis=0)
        phi_last = ScalarField(state.grid, data=field_data, label=f"Field {self.dim}")
        return FieldCollection(state.fields + [phi_last])

    def regularize_state(self, state: FieldCollection) -> float:
        """regularize the data portion of a scalar field

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                The fields describing the concentration distributions

        Returns:
            float: a measure for the corrections applied to the state
        """
        return self.f_local.regularize_state(state.data)  # type: ignore

    def free_energy(self, state: FieldCollection, t: float = 0) -> float:
        """evaluate the total free energy associated with the phase field

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                The fields describing the concentration distributions
            t (float):
                The current time point

        Returns:
            float: The total free energy (including the gradient terms)
            integrated over the entire grid.
        """
        self._check_field_type(state)

        # evaluate gradient terms
        gradients = [
            state[i].gradient(bc=self.bc_phi[i])  # type: ignore
            for i in range(self.dim)
        ]
        f_grad_integral: float = sum(
            self._kappa_reduced[i, j] / 2 * (gradients[i] @ gradients[j]).integral
            for i in range(self.dim)
            for j in range(self.dim)
        )

        # evaluate the local free energy
        f_local = ScalarField(state.grid, self.f_local.free_energy(state.data, t=t))
        return f_local.integral.real + f_grad_integral

    @property
    def free_energy_expression(self):
        """str: the mathematical expression giving the integrand of the free
        energy functional"""
        result = self.f_local.expression
        vs = self.f_local.variables
        for i in range(self.dim):
            for j in range(self.dim):
                if self._kappa_reduced[i, j] != 0:
                    if i == j:
                        field = f"grad({vs[i]})**2"
                    else:
                        field = f"grad({vs[i]}) * grad({vs[j]})"
                    result += f" + {self._kappa_reduced[i, j] / 2} * {field}"
        return result

    def chemical_potential(
        self, state: FieldCollection, t: float = 0
    ) -> FieldCollection:
        """return the (exchange) chemical potentials for a given state

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                The fields describing the concentration distributions
            t (float):
                The current time point

        Returns:
            :class:`~pde.fields.FieldCollection`:
            The (exchange) chemical potentials associated with the fields
        """
        self._check_field_type(state)

        # evaluate chemical potential
        mu = state.copy(label="chemical potential")
        mu.data[:] = self.f_local.chemical_potential(state.data, t=t)
        for i in range(self.dim):
            phi_i_lap = state[i].laplace(bc=self.bc_phi[i]).data  # type: ignore
            for j in range(self.dim):
                mu.data[j] -= self._kappa_reduced[i, j] * phi_i_lap
        return mu

    def reaction_rates(
        self, state: FieldCollection, mu: Optional[FieldCollection] = None, t: float = 0
    ) -> FieldCollection:
        """return the rate of change for each field due to chemical reactions

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                The fields describing the concentration distributions
            mu (:class:`~pde.fields.FieldCollection`, optional):
                The chemical potentials corresponding to the concentration fields . If
                omitted, the chemical potentials are calculated based on `state`.
            t (float):
                The current point in time

        Returns:
            :class:`~pde.fields.FieldCollection`:
            The reaction rates for all components.
        """
        self._check_field_type(state)

        if not self.reactions.present:
            # no reaction is given
            return state.copy(data=0)

        if mu is None:
            mu = self.chemical_potential(state, t=t)

        data = self.reactions.reaction_rates(state.data, mu.data, t)
        return state.copy(data=data, label="reaction fluxes")

    def determine_rhs_implementation_method(self, state: FieldCollection) -> str:
        """determines the method used for the implementation of the right hand side

        This method favors material conservation over correct implementation of the
        concentration-dependent mobility by choosing the method `split` if necessary.

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                The fields describing the concentration distributions

        Returns:
            str: an identifier determining the implementation method
        """
        method = str(self.parameters["rhs_implementation"])
        if method == "auto":
            mobility_model = self.parameters["mobility_model"]
            if mobility_model == "constant":
                method = "laplace"
            elif mobility_model == "kramer":
                if isinstance(state.grid, CartesianGridBase) and state.grid.dim <= 2:
                    method = "staggered"
                else:
                    method = "split"
            else:
                ValueError(f"Unknown mobility model `{mobility_model}`")
        return method

    def evolution_rate(  # type: ignore
        self,
        state: FieldCollection,
        t: float = 0,
    ) -> FieldCollection:
        """evaluate the right hand side of the PDE

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                The fields describing the concentration distributions
            t (float):
                The current time point

        Returns:
            :class:`~pde.fields.FieldCollection`:
            The evolution rate of each component
        """
        self._check_field_type(state)
        rhs = state.copy(label="evolution rate")

        # evaluate chemical potential
        mu = self.chemical_potential(state, t=t)

        # pick the correct implementation of the right hand side
        rhs_implementation = self.determine_rhs_implementation_method(state)
        if rhs_implementation == "laplace":
            # calculate the diffusive flux assuming constant mobilities

            # calculate the relevant derivatives
            laplace_mu_terms = [
                mu[i].laplace(bc=self.bc_mu[i]).data  # type: ignore
                for i in range(self.dim)
            ]

            assert self.parameters["mobility_model"] == "constant", "Need constant mob."
            for i in range(self.dim):
                rhs.data[i] = sum(
                    self.mobilities[i, j] * laplace_mu_terms[j] for j in range(self.dim)
                )

        elif rhs_implementation == "split":
            # calculate the diffusive flux assuming constant mobilities
            assert self.parameters["mobility_model"] == "kramer", "Need Kramer's model"
            # calculate the diffusive flux with Kramer's model for the mobilities
            rhs.data[:] = 0.0

            # calculate relevant derivatives
            gradient_phi_terms = [
                state[i].gradient(bc=self.bc_phi[i]).data  # type: ignore
                for i in range(self.dim)
            ]
            gradient_mu_terms = [
                mu[i].gradient(bc=self.bc_mu[i]).data  # type: ignore
                for i in range(self.dim)
            ]
            laplace_mu_terms = [
                mu[i].laplace(bc=self.bc_mu[i]).data  # type: ignore
                for i in range(self.dim)
            ]

            # add the fluxes
            for i in range(self.dim):  # calculate the change for each field i ...
                for j in range(self.dim):  # ... driven by all other fields j
                    mob = self.mobilities[i, j]
                    # terms with gradient of mu
                    if i == j:
                        f1 = (1 - state.data[j]) * gradient_phi_terms[i]
                    else:
                        f1 = -state.data[j] * gradient_phi_terms[i]
                    f1 -= state.data[i] * gradient_phi_terms[j]
                    rhs.data[i] += mob * np.einsum(
                        "k...,k...->...", f1, gradient_mu_terms[j]
                    )

                    # terms with Laplacian of mu
                    if i == j:
                        f2 = (1 - state.data[i]) * state.data[i]
                    else:
                        f2 = -state.data[i] * state.data[j]
                    rhs.data[i] += mob * f2 * laplace_mu_terms[j]

        elif rhs_implementation == "staggered":
            # calculate the diffusive flux assuming constant mobilities
            bc_ok = self.parameters["bc_mu"] in {"natural", "auto_periodic_neumann"}
            assert bc_ok, "Need simple boundary condition for staggered grid"
            assert isinstance(state.grid, CartesianGridBase), "Need Cartesian grid"

            from ..grids.operators.cartesian_staggered import (
                make_divergence_from_staggered_scipy,
                make_mc_flux_to_staggered_scipy,
            )

            mobility_model = self.parameters["mobility_model"]
            if mobility_model == "kramer":
                kramer = True
            elif mobility_model == "constant":
                kramer = False
            else:
                ValueError(f"Unknown mobility model `{mobility_model}`")

            get_flux_staggered = make_mc_flux_to_staggered_scipy(
                state.grid,
                num_comp=len(state),
                diffusivity=self.mobilities,
                kramer=kramer,
            )
            bcs = state.grid.get_boundary_conditions("auto_periodic_dirichlet")
            div_staggered = make_divergence_from_staggered_scipy(bcs)

            fluxes = get_flux_staggered(state.data, mu.data)
            for n, flux in enumerate(fluxes):
                div_staggered(flux, out=rhs[n].data)

        else:
            raise ValueError(
                f"Unknown method `{rhs_implementation}` for implementing the "
                "differential equation"
            )

        # add the chemical reaction
        if self.reactions.present:
            rhs += self.reactions.reaction_rates(state.data, mu=mu.data, t=t)

        return rhs

    def noise_realization(  # type: ignore
        self,
        state: FieldCollection,
        t: float = 0,
        label: str = "Noise realization",
    ) -> FieldCollection:
        """calculates a realization of the noise

        The boundary conditions are either no-flux or periodic, depending on the grid

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                The fields describing the concentration distributions
            t (float):
                The time point at which the evolution rate is determined
            label (str):
                The label for the returned field

        Returns:
            :class:`~pde.fields.FieldCollection`: containing the evolution rate. This
            could be directly used in an Euler stepping with time step `dt`.
        """
        # initialize the output
        self._check_field_type(state)
        noise = state.copy(data=0, label="noise realization")
        grid = state.grid

        # add diffusive noise respecting the correct mobility model
        if self.parameters["mobility_model"] == "constant":
            # calculate diffusive noise with constant mobilities
            if any(self.noise_diffusion):
                # draw uncorrelated fluctuations for all thermodynamic forces
                mu_grad = [VectorField.random_normal(grid) for i in range(self.dim)]
                mob_mat = _matrix_decomposition(self.mobilities)

                for i in range(self.dim):
                    if self.noise_diffusion[i]:
                        # calculate fluctuations of thermodynamic fluxes
                        flux = sum(mob_mat[i, j] * mu_grad[j] for j in range(self.dim))
                        flux_div = flux.divergence(bc="auto_periodic_dirichlet")  # type: ignore
                        noise[i] = self.noise_diffusion[i] * flux_div

        elif self.parameters["mobility_model"] == "kramer":
            # calculate diffusive noise with Kramer's mobility model
            if not np.allclose(self.mobilities, np.diag(np.diagonal(self.mobilities))):
                # mobility matrix is not diagonal
                raise NotImplementedError(
                    "Noise with Kramer's mobility model is only supported for diagonal "
                    "mobility matrices"
                )

            # calculate each diffusive flux independently
            for i in range(self.dim):
                if self.noise_diffusion[i] * self.mobilities[i, i]:
                    # calculate fluctuations of thermodynamic fluxes
                    factor = np.sqrt(self.mobilities[i, i] * state[i] * (1 - state[i]))
                    flux = factor * VectorField.random_normal(grid)
                    flux_div = flux.divergence(bc="auto_periodic_dirichlet")
                    noise[i] = self.noise_diffusion[i] * flux_div

        else:
            # other mobility models
            raise NotImplementedError(
                f"Noise for mobility model `{self.parameters['mobility_model']}` is "
                "not yet implemented"
            )

        # add reaction noise
        for i in range(self.dim):
            if self.noise_reaction[i]:
                noise[i] += ScalarField.random_normal(
                    grid, std=self.noise_reaction[i], scaling="physical"
                )

        noise.label = label
        return noise

    def _make_pde_rhs_numba_laplace(
        self, state: FieldCollection
    ) -> Callable[[np.ndarray, np.ndarray, float, np.ndarray], None]:
        """handle conservative part of the rhs of the PDE using laplace operators

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                An example for the state defining the grid and data types
        """
        assert self.parameters["mobility_model"] == "constant", "Need constant mobility"
        self._check_field_type(state)

        grid = state.grid
        laplace_mu = tuple(
            grid.get_operator("laplace", bc=self.bc_mu[i]) for i in range(self.dim)
        )

        num_comp = self.dim
        grid_shape = grid.shape
        arr_type = nb.typeof(np.empty((num_comp,) + grid.shape, dtype=np.double))

        # determine whether the laplace operator needs to be calculated
        mobilities = self.mobilities
        calc_laplace_mu = np.any(mobilities != 0, axis=0)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)

            @jit(nb.none(arr_type, arr_type, nb.double, arr_type))
            def pde_rhs_conserved(
                phi: np.ndarray, mu: np.ndarray, t: float, rhs: np.ndarray
            ) -> None:
                """ calculate the conservative part of the right hand side of PDE """
                data_lap_mu_n = np.empty(grid_shape)  # temporary array

                # calculate the effect of the diffusive fluxes
                for n in range(num_comp):
                    if calc_laplace_mu[n]:
                        laplace_mu[n](mu[n], out=data_lap_mu_n)
                        for m in range(num_comp):
                            if mobilities[m, n] != 0:
                                rhs[m] += mobilities[m, n] * data_lap_mu_n

        return pde_rhs_conserved  # type: ignore

    def _make_pde_rhs_numba_split(
        self, state: FieldCollection
    ) -> Callable[[np.ndarray, np.ndarray, float, np.ndarray], None]:
        """handle conservative part of the rhs of the PDE splitting the differential ops

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                An example for the state defining the grid and data types
        """
        assert self.parameters["mobility_model"] == "kramer", "Need Kramer's mobilities"
        self._check_field_type(state)

        grid = state.grid
        laplace_mu = tuple(
            grid.get_operator("laplace", bc=self.bc_mu[i]) for i in range(self.dim)
        )

        num_comp = self.dim
        grid_shape = grid.shape
        data_shape = (num_comp,) + grid.shape
        arr_type = nb.typeof(np.empty(data_shape, dtype=np.double))

        # determine whether the laplace operator needs to be calculated
        mobilities = self.mobilities

        # assume mobilities scaled according to Kramer's model
        space_dim = grid.dim
        gradient_full_shape = (num_comp, space_dim) + grid_shape
        gradient_one_shape = (space_dim,) + grid_shape
        grad_phi = tuple(
            grid.get_operator("gradient", bc=self.bc_phi[i]) for i in range(self.dim)
        )
        grad_mu = tuple(
            grid.get_operator("gradient", bc=self.bc_mu[i]) for i in range(self.dim)
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)

            @jit(nb.none(arr_type, arr_type, nb.double, arr_type))
            def pde_rhs_conserved(
                phi: np.ndarray, mu: np.ndarray, t: float, rhs: np.ndarray
            ) -> None:
                """ calculate the conservative part of the right hand side of PDE """
                # create temporary arrays
                data_lap_mu_j = np.empty(grid_shape)
                data_grad_phi = np.empty(gradient_full_shape)
                data_grad_mu_j = np.empty(gradient_one_shape)

                # calculate gradient terms
                for i in range(num_comp):
                    grad_phi[i](phi[i], out=data_grad_phi[i])

                # calculate the effect of the diffusive fluxes
                for j in range(num_comp):  # calculate the effect of all fields j ...
                    laplace_mu[j](mu[j], out=data_lap_mu_j)
                    grad_mu[j](mu[j], out=data_grad_mu_j)
                    for i in range(num_comp):  # ... onto the change for each field i
                        mob = mobilities[i, j]
                        # terms with gradient mu
                        if i == j:
                            f1 = (1 - phi[j]) * data_grad_phi[i]
                        else:
                            f1 = -phi[j] * data_grad_phi[i]
                        f1 -= phi[i] * data_grad_phi[j]
                        for k in range(space_dim):
                            rhs[i] += mob * f1[k, ...] * data_grad_mu_j[k, ...]

                        # terms with laplace mu
                        if i == j:
                            f2 = (1 - phi[i]) * phi[i]
                        else:
                            f2 = -phi[i] * phi[j]
                        rhs[i] += mob * f2 * data_lap_mu_j

        return pde_rhs_conserved  # type: ignore

    def _make_pde_rhs_numba_staggered(
        self, state: FieldCollection
    ) -> Callable[[np.ndarray, np.ndarray, float, np.ndarray], None]:
        """handle conservative part of the rhs of the PDE using a staggered grid

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                An example for the state defining the grid and data types
        """
        bc_ok = self.parameters["bc_mu"] in {"natural", "auto_periodic_neumann"}
        assert bc_ok, "Need simple boundary condition for staggered grid"
        assert isinstance(state.grid, CartesianGridBase), "Need Cartesian grid"
        self._check_field_type(state)

        num_comp = self.dim
        vec_data_shape = (num_comp, state.grid.dim) + state.grid.shape
        arr_type = nb.typeof(np.empty((num_comp,) + state.grid.shape, dtype=np.double))

        mobility_model = self.parameters["mobility_model"]
        if mobility_model == "kramer":
            kramer = True
        elif mobility_model == "constant":
            kramer = False
        else:
            ValueError(f"Unknown mobility model `{mobility_model}`")

        from ..grids.operators.cartesian_staggered import (
            make_divergence_from_staggered_numba,
            make_mc_flux_to_staggered_numba,
        )

        get_flux_staggered = make_mc_flux_to_staggered_numba(
            state.grid, num_comp=num_comp, diffusivity=self.mobilities, kramer=kramer
        )
        bcs = state.grid.get_boundary_conditions("auto_periodic_dirichlet")
        get_div_staggered = make_divergence_from_staggered_numba(bcs)

        @jit(nb.none(arr_type, arr_type, nb.double, arr_type))
        def pde_rhs_conserved(
            phi: np.ndarray, mu: np.ndarray, t: float, rhs: np.ndarray
        ) -> None:
            """ calculate the conservative part of the right hand side of PDE """
            fluxes = np.empty(vec_data_shape)  # temporary array
            get_flux_staggered(phi, mu, fluxes)
            for n in range(num_comp):
                get_div_staggered(fluxes[n, ...], rhs[n, ...])

        return pde_rhs_conserved  # type: ignore

    def _make_pde_rhs_numba(  # type: ignore
        self, state: FieldCollection
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        """create a compiled function evaluating the right hand side of the PDE

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                An example for the state defining the grid and data types

        Returns:
            A function with signature `(state_data, t)`, which can be called
            with an instance of :class:`~numpy.ndarray` of the state data and
            the time to obtained an instance of :class:`~numpy.ndarray` giving
            the evolution rate.
        """
        # check whether the state is reasonable
        self._check_field_type(state)

        # check whether the state is consistent with the free energy
        state_test = state.copy()
        self.regularize_state(state_test)
        if not np.allclose(state.data, state_test.data):
            self._logger.warning(
                "The initial state might violate constraints imposed by the free "
                "energy density. The resulting simulation might be invalid."
            )

        grid = state.grid
        num_comp = self.dim
        grid_shape = grid.shape
        data_shape = (self.dim,) + grid_shape
        arr_type = nb.typeof(np.empty(data_shape, dtype=np.double))
        signature = arr_type(arr_type, nb.double)

        # prepare the functions to calculate the chemical potential
        kappa_reduced = self._kappa_reduced
        mu_local = self.f_local.make_chemical_potential(backend="numba")
        laplace_phi = tuple(
            grid.get_operator("laplace", bc=self.bc_phi[i]) for i in range(self.dim)
        )
        # determine whether the laplace operator needs to be calculated
        calc_laplace_phi = np.any(kappa_reduced != 0, axis=1)

        # prepare the function calculating the diffusive fluxes
        rhs_implementation = self.determine_rhs_implementation_method(state)
        if rhs_implementation == "laplace":
            self._logger.info("Use constant mobility implementation")
            pde_rhs_conserved = self._make_pde_rhs_numba_laplace(state)
        elif rhs_implementation == "split":
            self._logger.info("Use split evaluation of the differential operators")
            pde_rhs_conserved = self._make_pde_rhs_numba_split(state)
        elif rhs_implementation == "staggered":
            self._logger.info("Use staggered grid for the differential operators")
            pde_rhs_conserved = self._make_pde_rhs_numba_staggered(state)
        else:
            raise ValueError(
                f"Unknown method `{rhs_implementation}` for implementing the "
                "differential equation"
            )

        # prepare the reaction term
        reactions_present = self.reactions.present
        apply_reaction = self.reactions.make_apply_reaction_rates_compiled(
            skip_last_component=True
        )

        @jit(signature)
        def pde_rhs(phi: np.ndarray, t: float) -> np.ndarray:
            """ calculate the right hand side of the PDE and return it """
            # create temporary arrays
            data_lap_phi_n = np.empty(grid_shape)
            rhs = np.zeros(data_shape)
            data_mu = np.empty(data_shape)

            # evaluate the chemical potential of all components
            mu_local(phi, t=t, out=data_mu)
            for n in range(num_comp):
                if calc_laplace_phi[n]:
                    laplace_phi[n](phi[n], out=data_lap_phi_n)
                    for m in range(num_comp):
                        data_mu[m] -= kappa_reduced[m, n] * data_lap_phi_n

            # apply the (conservative) diffusive fluxes
            pde_rhs_conserved(phi, data_mu, t, rhs)

            # add the chemical reaction
            if reactions_present:
                apply_reaction(phi, data_mu, t, rhs)
            return rhs

        return pde_rhs  # type: ignore

    def _make_step_splitting_numpy(
        self, phi: FieldCollection, max_error: float = 1e-8
    ) -> Callable[
        [np.ndarray, np.ndarray, float, float, bool], Tuple[int, float, float]
    ]:
        """return a function doing a single implicit step using numpy

        Args:
            phi (:class:`~pde.fields.FieldCollection`):
                An example for the state defining the grid and data types
            max_error (float):
                The maximal error, which determines when the implicit step is
                truncated.

        Returns:
            A function with signature
            `(phi_data, mu_data, t: float, dt: float, update_mu: bool=True)`, which
            solves the implicit equation for `phi-data` at time `t`, to advance the
            simulation by `dt`. If `update_mu == True`, the  chemical potential is
            calculated based on the concentration field given in `phi_data`
        """
        # check some pre-conditions for this algorithm
        assert not self.is_sde, "Energy splitting does not support noise"
        assert self.parameters["mobility_model"] == "constant", "Need constant mobility"
        self._check_field_type(phi)
        grid = phi.grid

        phi = phi.copy()  # will be changed in iteration
        mu = phi.copy(data=0, label="Chemical potential")
        kappa = self._kappa_reduced
        mobilities = self.mobilities

        # prepare boundary conditions
        get_op = grid.get_operator
        laplace_phi = [get_op("laplace", bc=self.bc_phi[i]) for i in range(self.dim)]
        laplace_mu = [get_op("laplace", bc=self.bc_mu[i]) for i in range(self.dim)]

        # pre-calculate data
        max_error2 = max_error ** 2
        dx_factor = np.sum(2 / grid.discretization ** 2)

        matrix = np.eye(2 * self.dim)
        matA = matrix[: self.dim, self.dim :]
        matB = matrix[self.dim :, : self.dim]
        vector = np.empty(2 * self.dim)
        vecA, vecB = vector[: self.dim], vector[self.dim :]

        # get functions to express the local part of the chemical potential
        try:
            make_mu_split = self.f_local.make_chemical_potential_split
        except (AttributeError, NotImplementedError):
            raise NotImplementedError(
                f"Free energy {self.f_local} does not support energy splitting"
            )
        else:
            mu_local_ex, mu_local_im, mu_local_im_diff = make_mu_split("numpy")

        phi_lap = np.empty((self.dim,) + grid.shape)
        mu_lap = np.empty((self.dim,) + grid.shape)

        def step_splitting(
            phi_data: np.ndarray,
            mu_data: np.ndarray,
            t: float,
            dt: float,
            update_mu: bool = True,
        ) -> Tuple[int, float, float]:
            """evaluate the implicit step given state data without the grid.

            Args:
                phi_data (:class:`~numpy.ndarray`): The phase field
                mu_data (:class:`~numpy.ndarray`): Associated chemical potential field
                t (float): Current time
                dt (float): Time step
                update_mu (bool): Flag indicating whether `mu_data` should be updated

            Returns:
                tuple (iterations, correction, error): The number of iterations until
                convergence, the regularization correction applied and the remaining
                residual.
            """
            phi.data = phi_data  # set new data
            if update_mu:
                # update chemical potential
                mu.data = self.chemical_potential(phi).data
            else:
                mu.data = mu_data
            phi_lhs = phi.copy()

            # calculate explicit terms
            mu_lhs = mu_local_ex(phi.data, t=t)
            if self.reactions.present:
                # treat reactions explicitly
                reaction_rates = self.reactions.reaction_rates(phi.data, mu.data, t)
                phi_lhs += dt * reaction_rates

            # setup the matrices for the linear system
            matA[:] = dt * dx_factor * mobilities

            # solve implicit part iteratively
            iterations = 0
            correction = 0.0
            while True:
                iterations += 1

                # calculate things necessary for error and implicit stepping
                for i in range(self.dim):
                    laplace_phi[i](phi.data[i], out=phi_lap[i])
                    laplace_mu[i](mu.data[i], out=mu_lap[i])
                mu_im = mu_local_im(phi.data, t=t)

                # remove the central part from the laplace estimate
                phi_lap_nn = phi_lap + dx_factor * phi.data
                mu_lap_nn = mu_lap + dx_factor * mu.data

                # setup linearized system for the implicit step for each grid point
                for loc in np.ndindex(*grid.shape):
                    i = (slice(None),) + loc  # index for all components at one point
                    matB[:] = -dx_factor * kappa - mu_local_im_diff(phi.data[i], t=t)
                    vecA[:] = phi_lhs.data[i] + dt * mobilities @ mu_lap_nn[i]
                    vecB[:] = (
                        mu_lhs[i]
                        + mu_im[i]
                        - mu_local_im_diff(phi.data[i], t=t) @ phi.data[i]
                        - kappa @ phi_lap_nn[i]
                    )

                    # solve linearized system
                    sol = np.linalg.solve(matrix, vector)
                    phi.data[i] = sol[: self.dim]
                    mu.data[i] = sol[self.dim :]

                # regularize the state to avoid problems with logarithms
                correction += self.regularize_state(phi)

                # calculate how close we are to the solution
                phi_rhs = phi.data - dt * np.tensordot(mobilities, mu_lap, axes=1)
                mu_rhs = mu.data - mu_im + np.tensordot(kappa, phi_lap, axes=1)
                error2_phi = np.sum((phi_lhs.data - phi_rhs) ** 2)
                error2_mu = np.sum((mu_lhs - mu_rhs) ** 2)
                error2 = error2_phi + error2_mu
                if error2 < max_error2:
                    break  # return once we converged

                if iterations > self.parameters["implicit_iterations_max"]:
                    raise ImplicitConvergenceError(
                        "Implicit step did not converge after "
                        f"{self.parameters['implicit_iterations_max']} steps. "
                        f"Residual: {np.sqrt(error2)}"
                    )

            # write result back to the input fields
            phi_data[:] = phi.data
            mu_data[:] = mu.data
            return iterations, correction, np.sqrt(error2)

        return step_splitting

    def _make_step_splitting_numba(
        self, phi: FieldCollection, max_error: float = 1e-8
    ) -> Callable[
        [np.ndarray, np.ndarray, float, float, Optional[bool]], Tuple[int, float, float]
    ]:
        """return a function doing a single implicit step using numba

        Args:
            phi (:class:`~pde.fields.FieldCollection`):
                An example for the state defining the grid and data types
            max_error (float):
                The maximal error, which determines when the implicit step is
                truncated.

        Returns:
            A function with signature
            `(phi_data, mu_data, t: float, dt: float, update_mu: bool=True)`, which
            solves the implicit equation for `phi-data` at time `t`, to advance the
            simulation by `dt`. If `update_mu == True`, the  chemical potential is
            calculated based on the concentration field given in `phi_data`
        """
        # check some pre-conditions for this algorithm
        assert self.parameters["mobility_model"] == "constant", "Need constant mobility"
        assert self.is_sde == False, "Energy splitting does not support noise"
        self._check_field_type(phi)
        grid = phi.grid

        kappa = self._kappa_reduced
        mobilities = self.mobilities
        regularize = jit(self.f_local.make_state_regularizer(phi))

        # prepare boundary conditions
        get_laplace = lambda bc: grid.get_operator("laplace", bc=bc)
        laplace_phi = tuple(get_laplace(self.bc_phi[i]) for i in range(self.dim))
        laplace_mu = tuple(get_laplace(self.bc_mu[i]) for i in range(self.dim))

        # pre-calculate data
        max_error2 = max_error ** 2
        dx_factor = np.sum(2 / grid.discretization ** 2)
        implicit_iterations_max = self.parameters["implicit_iterations_max"]
        iterations_max_err = (
            f"Implicit step did not converge after {implicit_iterations_max:d} steps"
        )

        # prepare the reaction term
        reactions_present = self.reactions.present
        apply_reaction = self.reactions.make_apply_reaction_rates_compiled(
            skip_last_component=True
        )

        # get functions to express the local part of the chemical potential
        try:
            make_mu_split = self.f_local.make_chemical_potential_split
        except (AttributeError, NotImplementedError):
            raise NotImplementedError(
                f"Free energy {self.f_local} does not support energy splitting"
            )
        else:
            mu_local_ex, mu_local_im, mu_local_im_diff = make_mu_split("numba")

        dim = self.dim
        shape = (dim,) + grid.shape
        size = int(np.product(grid.shape))
        shape_flat = (dim, size)

        # determine the signature
        arr_type = nb.typeof(np.empty(shape, dtype=np.double))
        ret_type = nb.types.Tuple((nb.int64, nb.float64, nb.float64))
        signature = ret_type(arr_type, arr_type, nb.double, nb.double, nb.boolean)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
            warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)

            @jit(signature)
            def step_splitting(
                phi: np.ndarray,
                mu: np.ndarray,
                t: float,
                dt: float,
                update_mu: bool,
            ) -> Tuple[int, float, float]:
                """evaluate the implicit step given state data without the grid.

                Args:
                    phi_data (:class:`~numpy.ndarray`): The phase field
                    mu_data (:class:`~numpy.ndarray`): Associated chemical potential field
                    t (float): Current time
                    dt (float): Time step
                    update_mu (bool): Flag indicating whether `mu_data` should be updated

                Returns:
                    tuple (iterations, correction, error): The number of iterations until
                    convergence, the regularization correction applied and the remaining
                    residual.
                """
                # allocate temporary arrays
                phi_lap = np.empty(shape, dtype=np.double)
                mu_lap = np.empty(shape, dtype=np.double)

                # create flat views of these arrays
                phi_flat = phi.reshape(shape_flat)
                mu_flat = mu.reshape(shape_flat)
                phi_lap_flat = phi_lap.reshape(shape_flat)
                mu_lap_flat = mu_lap.reshape(shape_flat)

                # allocate flat arrays
                phi_lhs = phi.copy()  # preserve for comparison
                phi_lhs_flat = phi_lhs.reshape(shape_flat)
                mu_lhs_flat = np.empty(shape_flat, dtype=np.double)

                # calculate explicit term of the chemical potential
                for k in nb.prange(size):
                    mu_local_ex(phi_lhs_flat[:, k], t=t, out=mu_lhs_flat[:, k])

                if update_mu:
                    # update chemical potential
                    for i in nb.prange(dim):
                        laplace_phi[i](phi[i], out=phi_lap[i])

                    for k in nb.prange(size):
                        mu_local_im(phi_flat[:, k], t=t, out=mu_flat[:, k])
                        mu_flat[:, k] += mu_lhs_flat[:, k] - kappa @ phi_lap_flat[:, k]

                # apply chemical reactions
                if reactions_present:
                    apply_reaction(phi, mu, t, phi_lhs, prefactor=dt)

                # prepare the linear system
                matrix = np.eye(2 * dim)
                matrix[:dim, dim:] = dt * dx_factor * mobilities
                vector = np.empty(2 * dim)

                # solve implicit part iteratively
                iterations, correction = 0, 0.0
                while True:
                    iterations += 1

                    # calculate the laplace operators of both fields
                    for i in nb.prange(dim):
                        laplace_phi[i](phi[i], out=phi_lap[i])
                        laplace_mu[i](mu[i], out=mu_lap[i])

                    # do a Gauss-Seidel iteration at each position
                    error2 = 0.0  # total squared error
                    for k in range(size):
                        # remove the central part from the laplace estimate
                        phi_lap_nn = phi_lap_flat[:, k] + dx_factor * phi_flat[:, k]
                        mu_lap_nn = mu_lap_flat[:, k] + dx_factor * mu_flat[:, k]
                        # pre-calculate some chemical potentials
                        mu_local_im_k = mu_local_im(phi_flat[:, k], t=t)
                        mu_local_im_diff_k = mu_local_im_diff(phi_flat[:, k], t=t)

                        # prepare the vector and matrix for the linearized system
                        for i in range(dim):
                            vector[i] = (
                                phi_lhs_flat[i, k] + dt * mobilities[i, :] @ mu_lap_nn
                            )
                            vector[dim + i] = (
                                mu_lhs_flat[i, k]
                                + mu_local_im_k[i]
                                - mu_local_im_diff_k[i, :] @ phi_flat[:, k]
                                - kappa[i, :] @ phi_lap_nn
                            )
                            for j in range(dim):
                                matrix[dim + i, j] = (
                                    -dx_factor * kappa[i, j] - mu_local_im_diff_k[i, j]
                                )

                        # solve linearized system (Gauss-Seidel iteration)
                        sol = np.linalg.solve(matrix, vector)
                        phi_flat[:, k] = sol[:dim]
                        mu_flat[:, k] = sol[dim:]

                        # correct the phi field to lie within the specified bounds
                        correction += regularize(phi_flat[:, k])

                        # calculate how close we are to the solution
                        mu_local_im_k = mu_local_im(phi_flat[:, k], t=t)
                        for i in range(dim):
                            # calculate error in phi field
                            error2 += (
                                phi_flat[i, k]
                                - dt * mobilities[i, :] @ mu_lap_flat[:, k]
                                - phi_lhs_flat[i, k]
                            ) ** 2

                            # calculate error in mu field
                            error2 += (
                                mu_flat[i, k]
                                - mu_local_im_k[i]
                                + kappa[i, :] @ phi_lap_flat[:, k]
                                - mu_lhs_flat[i, k]
                            ) ** 2

                    if error2 < max_error2:
                        break  # return once we converged

                    if iterations > implicit_iterations_max:
                        raise ImplicitConvergenceError(iterations_max_err)

                return iterations, correction, np.sqrt(error2)

        return step_splitting  # type: ignore

    def _make_noise_realization_numba(  # type: ignore
        self, state: FieldCollection
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        """return a function calculating a realization of the noise

        The boundary conditions are either no-flux or periodic, depending on the grid

        Args:
            state (:class:`~pde.fields.collection.FieldCollection`):
                An example for the state from which the grid and other
                information can be extracted

        Returns:
            A function with signature `(state_data, t: float)`. `state_data` is
            a :class:`~pde.fields.collection.FieldCollection` for the `numpy`
            backend and a numpy.array for the `numba` backend. The return value
            of the function has the same type as `state_data`.
        """
        # extract parameters
        dim = self.dim
        grid = state.grid
        state_shape = (dim,) + grid.shape
        grid_dim = grid.dim
        grid_size = np.product(grid.shape)
        grad_mu_shape = (dim, grid_dim) + grid.shape
        vec_shape = (grid_dim,) + grid.shape
        arr_type = nb.typeof(np.empty(state_shape, dtype=np.double))
        signature = arr_type(arr_type, nb.double)

        # extract the noise strengths
        noise_strength_diff = self.noise_diffusion
        noise_strength_react = self.noise_reaction
        noise_diff = any(self.noise_diffusion)
        noise_react = any(self.noise_reaction)
        if noise_diff and self.parameters["mobility_model"] != "constant":
            raise NotImplementedError(
                f"Noise for mobility model `{self.parameters['mobility_model']}` is "
                "not yet implemented"
            )

        # create a compiled function necessary for diffusive noise
        mobility_mat = _matrix_decomposition(self.mobilities)
        divergence_flux = grid.get_operator("divergence", bc="auto_periodic_dirichlet")
        cell_volume = state.grid.make_cell_volume_compiled(flat_index=True)

        @jit(signature)
        def noise_realization(state_data: np.ndarray, t: float) -> np.ndarray:
            """ calculate noise realization associated with `state_data` """
            noise_grad_mu = np.empty(grad_mu_shape, np.double)
            noise_flux = np.empty(vec_shape, np.double)
            noise_data = np.zeros(state_shape, np.double)  # array to store result

            if noise_diff:
                # determine the fluctuations associated with gradients of mu
                for i in nb.prange(grid_size):  # iterate space
                    scale = 1 / np.sqrt(cell_volume(i))
                    for f_id in range(dim):  # iterate fields
                        for n in range(grid_dim):  # iterate vector components
                            noise_grad_mu[f_id, n].flat[i] = scale * np.random.randn()

                for f_id in range(dim):  # iterate fields
                    noise_strength = noise_strength_diff[f_id]
                    if noise_strength != 0:
                        # add contribution from diffusive fluxes for each field
                        for n in range(grid_dim):  # iterate vector components
                            for i in nb.prange(grid_size):  # iterate space
                                noise_flux[n].flat[i] = 0
                                for f2 in range(dim):  # iterate fields
                                    noise_flux[n].flat[i] += (
                                        noise_strength
                                        * mobility_mat[f_id, f2]
                                        * noise_grad_mu[f2, n].flat[i]
                                    )
                        divergence_flux(noise_flux, out=noise_data[f_id])

            # add reaction noise
            if noise_react:
                for i in nb.prange(grid_size):  # iterate space
                    cell_scaling = 1 / np.sqrt(cell_volume(i))
                    for f_id in range(dim):  # iterate fields
                        if noise_strength_react[f_id] != 0:
                            noise_data[f_id].flat[i] += (
                                cell_scaling
                                * noise_strength_react[f_id]
                                * np.random.randn()
                            )

            return noise_data

        return noise_realization  # type: ignore

    def make_modify_after_step(self, state: FieldBase) -> Callable[[np.ndarray], float]:
        """returns a function that can be called to modify a state

        This function is applied to the state after each integration step when
        an explicit stepper is used. The default behavior is to regularize the state
        when the parameter `regularize_after_step` is True.

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other
                information can be extracted

        Returns:
            Function that can be applied to a state to modify it and which
            returns a measure for the corrections applied to the state
        """
        if self.parameters["regularize_after_step"]:
            return self.f_local.make_state_regularizer(state)  # type: ignore
        else:
            return super().make_modify_after_step(state)
