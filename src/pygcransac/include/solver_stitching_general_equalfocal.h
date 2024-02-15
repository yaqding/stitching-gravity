// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include <Eigen/Eigen>
#include "solver_engine.h"
#include "fundamental_estimator.h"
#include "/usr/local/include/eigen3/unsupported/Eigen/Polynomials"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class StitchingGeneralEqual : public SolverEngine
			{
			public:
				StitchingGeneralEqual() : gravity_source(Eigen::Matrix3d::Identity()),
										gravity_destination(Eigen::Matrix3d::Identity())
				{
				}

				~StitchingGeneralEqual()
				{
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 2;
				}

				static constexpr size_t maximumSolutions()
				{
					return 3;
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return true;
				}

				static constexpr bool needsGravity()
				{
					return true;
				}

				static constexpr char * getName()
				{
					return "General Equal Focal (2PC)";
				}

				void setGravity(const Eigen::Matrix3d &gravity_source_,
								const Eigen::Matrix3d &gravity_destination_)
				{
					gravity_source = gravity_source_;
					gravity_destination = gravity_destination_;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat &data_,					 // The set of data points
					const size_t *sample_,					 // The sample used for the estimation
					size_t sample_number_,					 // The size of the sample
					std::vector<Model> &models_,			 // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point

			protected:
				Eigen::Matrix3d gravity_source;
				Eigen::Matrix3d gravity_destination;
			};

			OLGA_INLINE bool StitchingGeneralEqual::estimateModel(
				const cv::Mat &data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				using namespace Eigen;

				// Building the coefficient matrices
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const size_t cols = data_.cols;
				const size_t pointNumber = sample_number_; // 1

				// this solver is unstable, the points need to be normalized, maybe it's better to normalize points for all the solvers
				double scale_factor = 1000;
				const double
					&u1 = data_.at<double>(sample_[0], 0)/scale_factor,
					&v1 = data_.at<double>(sample_[0], 1)/scale_factor,
					&u2 = data_.at<double>(sample_[0], 2)/scale_factor,
					&v2 = data_.at<double>(sample_[0], 3)/scale_factor,
					&u3 = data_.at<double>(sample_[1], 0)/scale_factor,
					&v3 = data_.at<double>(sample_[1], 1)/scale_factor,
					&u4 = data_.at<double>(sample_[1], 2)/scale_factor,
					&v4 = data_.at<double>(sample_[1], 3)/scale_factor;

				double a12 = u1 * u3 + v1 * v3;
				double a1 = u1 * u1 + v1 * v1;
				double a2 = u3 * u3 + v3 * v3;

				double b12 = u2 * u4 + v2 * v4;
				double b1 = u2 * u2 + v2 * v2;
				double b2 = u4 * u4 + v4 * v4;

				double c0 = b1 * b2 * a12 * a12 - a1 * a2 * b12 * b12;
				double c1 = a12 * a12 * b1 - a1 * b12 * b12 - a2 * b12 * b12 + a12 * a12 * b2 - 2 * a1 * a2 * b12 + 2 * a12 * b1 * b2;
				double c2 = 2 * a12 * b1 - 2 * a1 * b12 - a1 * a2 - 2 * a2 * b12 + 2 * a12 * b2 + b1 * b2 + a12 * a12 - b12 * b12;
				double c3 = 2 * a12 - a2 - a1 + b1 + b2 - 2 * b12;

				MatrixXd AM(3, 3);
				AM << 0, 1, 0,
					0, 0, 1,
					-c0 / c3, -c1 / c3, -c2 / c3;

				EigenSolver<MatrixXd> es(AM);
				ArrayXcd D = es.eigenvalues(); // focal length^2 - f^2

				for (size_t k = 0; k < 3; ++k)
				{
					if (D(k).imag() > std::numeric_limits<double>::epsilon() || D(k).real() < 0)
						continue;

					double focal_n = sqrt(D(k).real());
					Eigen::Vector3d u11(u1 / focal_n, v1 / focal_n, 1);
					Eigen::Vector3d u21(u2 / focal_n, v2 / focal_n, 1);

					Eigen::Vector3d u12(u3 / focal_n, v3 / focal_n, 1);
					Eigen::Vector3d u22(u4 / focal_n, v4 / focal_n, 1);

					u11.normalize();
					u21.normalize();
					u12.normalize();
					u22.normalize();

					Eigen::Matrix3d C = u21 * u11.transpose() + u22 * u12.transpose();
					JacobiSVD<MatrixXd> svd(C, ComputeFullU | ComputeFullV);
					Eigen::Matrix3d U = svd.matrixU();
					Eigen::Matrix3d V = svd.matrixV().transpose();

					Eigen::Matrix3d S;
					S << 1, 0, 0,
						0, 1, 0,
						0, 0, ((U * V).determinant() > 0) ? 1 : -1;

					Eigen::Matrix3d R = U * S * V;
					Eigen::Matrix3d K;
					K << focal_n * scale_factor, 0, 0,
						0, focal_n * scale_factor, 0,
						0, 0, 1;
					Eigen::Matrix3d H = K * R * K.inverse();

					Homography model;
					model.descriptor = Eigen::MatrixXd(3, 4);
					model.descriptor.block<3,3>(0,0) = H;
					model.descriptor(2, 3) = focal_n * scale_factor;
					models_.push_back(model);
				}

				return models_.size() > 0;
			}

		} // namespace solver
	}	  // namespace estimator
} // namespace gcransac
