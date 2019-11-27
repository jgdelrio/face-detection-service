import pytest
from hamcrest import *
from src.face_detection import *
from models.definitions import Model


GET_MODEL_TEST_DATA = ["rfb", "slim"]


@pytest.mark.parametrize("data_in", GET_MODEL_TEST_DATA)
def test_get_model_reference(data_in):
    expected_properties = ('name', 'location', 'ref', 'version', 'speed', 'classes', 'description', 'aliases')
    output = get_model_reference("rfb")

    assert(isinstance(output, Model))
    assert_that(output, all_of(*[has_property(prop) for prop in expected_properties]))


def test_get_model_reference_with_wrong_model():
    assert_that(calling(get_model_reference).with_args("non_existing_model"), raises(ValueError))


def test_get_model_reference_with_none():
    assert_that(calling(get_model_reference).with_args(None), raises(ValueError))
