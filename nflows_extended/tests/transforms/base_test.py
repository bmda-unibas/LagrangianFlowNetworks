"""Tests for the basic transform definitions."""
import unittest

import numpy as np
import torch

from enflows.transforms import base, standard, ConditionalTransform, InverseTransform
from tests.transforms.transform_test import TransformTest


class CompositeTransformTest(TransformTest):

    def test_forward(self):
        batch_size = 10
        shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *shape)
        transforms = [
            standard.AffineScalarTransform(scale=2.0),
            standard.IdentityTransform(),
            standard.AffineScalarTransform(scale=0.25),
        ]
        composite = base.CompositeTransform(transforms)
        reference = standard.AffineScalarTransform(scale=0.5)
        outputs, logabsdet = composite(inputs)
        outputs_ref, logabsdet_ref = reference(inputs)
        self.assert_tensor_is_good(outputs, [batch_size] + shape)
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.assertEqual(outputs, outputs_ref)
        self.assertEqual(logabsdet, logabsdet_ref)

    def test_inverse(self):
        batch_size = 10
        shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *shape)
        transforms = [
            standard.AffineScalarTransform(scale=2.0),
            standard.IdentityTransform(),
            standard.AffineScalarTransform(scale=0.25),
        ]
        composite = base.CompositeTransform(transforms)
        reference = standard.AffineScalarTransform(scale=0.5)
        outputs, logabsdet = composite.inverse(inputs)
        outputs_ref, logabsdet_ref = reference.inverse(inputs)
        self.assert_tensor_is_good(outputs, [batch_size] + shape)
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.assertEqual(outputs, outputs_ref)
        self.assertEqual(logabsdet, logabsdet_ref)


class MultiscaleCompositeTransformTest(TransformTest):
    def create_transform(self, shape, split_dim=1):
        mct = base.MultiscaleCompositeTransform(num_transforms=4, split_dim=split_dim)
        for transform in [
            standard.AffineScalarTransform(scale=2.0),
            standard.AffineScalarTransform(scale=4.0),
            standard.AffineScalarTransform(scale=0.5),
            standard.AffineScalarTransform(scale=0.25),
        ]:
            shape = mct.add_transform(transform, shape)

        return mct

    def test_forward(self):
        batch_size = 5
        for shape in [(32, 4, 4), (64,), (65,)]:
            with self.subTest(shape=shape):
                inputs = torch.ones(batch_size, *shape)
                transform = self.create_transform(shape)
                outputs, logabsdet = transform(inputs)
                self.assert_tensor_is_good(outputs, [batch_size] + [np.prod(shape)])
                self.assert_tensor_is_good(logabsdet, [batch_size])

    def test_forward_bad_shape(self):
        shape = (8,)
        with self.assertRaises(ValueError):
            transform = self.create_transform(shape)

    def test_forward_bad_split_dim(self):
        batch_size = 5
        shape = [32]
        inputs = torch.randn(batch_size, *shape)
        with self.assertRaises(ValueError):
            transform = self.create_transform(shape, split_dim=2)

    def test_inverse_not_flat(self):
        batch_size = 5
        shape = [32, 4, 4]
        inputs = torch.randn(batch_size, *shape)
        transform = self.create_transform(shape)
        with self.assertRaises(ValueError):
            transform.inverse(inputs)

    def test_forward_inverse_are_consistent(self):
        batch_size = 5
        for shape in [(32, 4, 4), (64,), (65,), (21,)]:
            with self.subTest(shape=shape):
                transform = self.create_transform(shape)
                inputs = torch.randn(batch_size, *shape).view(batch_size, -1)
                self.assert_forward_inverse_are_consistent(InverseTransform(transform), inputs)


class InverseTransformTest(TransformTest):
    def test_forward(self):
        batch_size = 10
        shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *shape)
        transform = base.InverseTransform(standard.AffineScalarTransform(scale=2.0))
        reference = standard.AffineScalarTransform(scale=0.5)
        outputs, logabsdet = transform(inputs)
        outputs_ref, logabsdet_ref = reference(inputs)
        self.assert_tensor_is_good(outputs, [batch_size] + shape)
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.assertEqual(outputs, outputs_ref)
        self.assertEqual(logabsdet, logabsdet_ref)

    def test_inverse(self):
        batch_size = 10
        shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *shape)
        transform = base.InverseTransform(standard.AffineScalarTransform(scale=2.0))
        reference = standard.AffineScalarTransform(scale=0.5)
        outputs, logabsdet = transform.inverse(inputs)
        outputs_ref, logabsdet_ref = reference.inverse(inputs)
        self.assert_tensor_is_good(outputs, [batch_size] + shape)
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.assertEqual(outputs, outputs_ref)
        self.assertEqual(logabsdet, logabsdet_ref)


class ConditionalTransformTest(TransformTest):
    def setUp(self):
        self.features = 3
        self.batch_size = 10

        self.random_input = torch.randn((self.batch_size, self.features))

        self.transform = ConditionalTransform(features=self.features, conditional_net=torch.nn.Identity())

    def test_no_condition(self):
        with self.assertRaises(expected_exception=TypeError) as cm:
            self.transform.forward(self.random_input)

        with self.assertRaises(expected_exception=TypeError) as cm:
            self.transform.inverse(self.random_input)


if __name__ == "__main__":
    unittest.main()
