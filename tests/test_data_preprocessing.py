import numpy as np
import pytest
from hamcrest import *
from src.data_preprocessing import *


# @pytest.mark.parametrize("data_in", GET_MODEL_REFERENCE_DATA)
def test_intersect():
    box_a = np.array([[100, 100], [400, 400]])
    box_b = np.array([[200, 200], [500, 500]])
    output = intersect(box_a, box_b)
    expected_properties = ('name', 'location', 'ref', 'version', 'speed', 'classes', 'description', 'aliases')
#     output = get_model_reference("rfb")
#
#     assert(isinstance(output, Model))
#     assert_that(output, all_of(*[has_property(prop) for prop in expected_properties]))


if __name__ == "__main__":
    test_intersect()
