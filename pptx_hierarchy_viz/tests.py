
import pptx
from pptx.enum.shapes import MSO_SHAPE

def test_add_animation_single_shape():
    presentation = Presentation('path/to/presentation.pptx')
    slide = presentation.slides[0]
    shape = slide.shapes[0]
    add_animation(slide, shape)
    assert len(shape.animations) == 1

def test_add_animation_group_shape():
    presentation = Presentation('path/to/presentation.pptx')
    slide = presentation.slides[0]
    group_shape = slide.shapes.add_group()
    shape_1 = group_shape.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, 100, 100)
    shape_2 = group_shape.shapes.add_shape(MSO_SHAPE.OVAL, 100, 100, 200, 200)
    add_animation(slide, group_shape)
    assert len(shape_1.animations) == 1
    assert len(shape_2.animations) == 1
