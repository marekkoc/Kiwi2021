#!/usr/bin/env python

import argparse

import itk
from distutils.version import StrictVersion as VS

# Segment Blood Vessels With Multi-Scale Hessian-Based Measure
# https://itk.org/ITKExamples/src/Nonunit/Review/SegmentBloodVesselsWithMultiScaleHessianBasedMeasure/Documentation.html


if VS(itk.Version.GetITKVersion()) < VS("5.0.0"):
    print("ITK 5.0.0 or newer is required.")
    sys.exit(1)

parser = argparse.ArgumentParser(
    description="Segment blood vessels with multi-scale Hessian-based measure."
)
parser.add_argument("input_image")
parser.add_argument("output_image")
parser.add_argument("--sigma_minimum", type=float, default=1.0)
parser.add_argument("--sigma_maximum", type=float, default=10.0)
parser.add_argument("--number_of_sigma_steps", type=int, default=10)
args = parser.parse_args()

input_image = itk.imread(args.input_image, itk.F)

ImageType = type(input_image)
Dimension = input_image.GetImageDimension()
HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]
HessianImageType = itk.Image[HessianPixelType, Dimension]

objectness_filter = itk.HessianToObjectnessMeasureImageFilter[
    HessianImageType, ImageType
].New()
objectness_filter.SetBrightObject(False)
objectness_filter.SetScaleObjectnessMeasure(False)
objectness_filter.SetAlpha(0.5)
objectness_filter.SetBeta(1.0)
objectness_filter.SetGamma(5.0)

#objectness_filter.SetAlpha(0.5)
#objectness_filter.SetBeta(1.0)
#objectness_filter.SetGamma(5.0)

multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[
    ImageType, HessianImageType, ImageType
].New()
multi_scale_filter.SetInput(input_image)
multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
multi_scale_filter.SetSigmaStepMethodToLogarithmic()
multi_scale_filter.SetSigmaMinimum(args.sigma_minimum)
multi_scale_filter.SetSigmaMaximum(args.sigma_maximum)
multi_scale_filter.SetNumberOfSigmaSteps(args.number_of_sigma_steps)

OutputPixelType = itk.UC
OutputImageType = itk.Image[OutputPixelType, Dimension]

rescale_filter = itk.RescaleIntensityImageFilter[ImageType, OutputImageType].New()
rescale_filter.SetInput(multi_scale_filter)

itk.imwrite(rescale_filter.GetOutput(), args.output_image)
