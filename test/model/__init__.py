from mxlpy import Model
from .derived_quantities import include_derived_quantities
from .rates import include_rates

__all__ = [
    'test'
]

def test() -> Model:
    m = Model()

    m.add_parameters(
        {

        }
    )

    m.add_variables(
        {

        }
    )

    m = include_derived_quantities(m)
    m = include_rates(m)

    return m
