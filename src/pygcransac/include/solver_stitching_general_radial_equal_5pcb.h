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
			class StitchingGeneralRadialEqual5pc : public SolverEngine
			{
			public:
				StitchingGeneralRadialEqual5pc() : gravity_source(Eigen::Matrix3d::Identity()),
												   gravity_destination(Eigen::Matrix3d::Identity())
				{
				}

				~StitchingGeneralRadialEqual5pc()
				{
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 10;
				}

				static constexpr size_t maximumSolutions()
				{
					return 6;
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return true;
				}

				static constexpr bool needsGravity()
				{
					return false;
				}

				static constexpr char *getName()
				{
					return "General Equal Radial (5PC)";
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

				Eigen::Matrix3d K1, K2;

			protected:
				Eigen::Matrix3d gravity_source;
				Eigen::Matrix3d gravity_destination;
			};

			OLGA_INLINE bool StitchingGeneralRadialEqual5pc::estimateModel(
				const cv::Mat &data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				using namespace Eigen;

				Eigen::MatrixXd X(sample_number_, 3),
					U(sample_number_, 3);

				MatrixXd C1 = MatrixXd::Zero(2 * sample_number_, 9);
				MatrixXd C2 = MatrixXd::Zero(2 * sample_number_, 5);
				MatrixXd C3 = MatrixXd::Zero(2 * sample_number_, 1);

				for (size_t sampleIdx = 0; sampleIdx < sample_number_; ++sampleIdx)
				{
					const size_t &pointIdx = sample_[sampleIdx];

					X(sampleIdx, 0) = data_.at<double>(pointIdx, 0);
					X(sampleIdx, 1) = data_.at<double>(pointIdx, 1);
					X(sampleIdx, 2) = X(sampleIdx, 0) * X(sampleIdx, 0) + X(sampleIdx, 1) * X(sampleIdx, 1);
					U(sampleIdx, 0) = data_.at<double>(pointIdx, 2);
					U(sampleIdx, 1) = data_.at<double>(pointIdx, 3);
					U(sampleIdx, 2) = U(sampleIdx, 0) * U(sampleIdx, 0) + U(sampleIdx, 1) * U(sampleIdx, 1);

					C1.row(sampleIdx * 2) << U(sampleIdx, 0), U(sampleIdx, 1), 1, 0, 0, 0, -X(sampleIdx, 0) * U(sampleIdx, 0), -X(sampleIdx, 0) * U(sampleIdx, 1), -X(sampleIdx, 0);
					C1.row(sampleIdx * 2 + 1) << -X(sampleIdx, 1) * U(sampleIdx, 0), -X(sampleIdx, 1) * U(sampleIdx, 1), -X(sampleIdx, 1), X(sampleIdx, 0) * U(sampleIdx, 0), X(sampleIdx, 0) * U(sampleIdx, 1), X(sampleIdx, 0), 0, 0, 0;

					C2.row(sampleIdx * 2) << X(sampleIdx, 2) * U(sampleIdx, 0), X(sampleIdx, 2) * U(sampleIdx, 1), X(sampleIdx, 2) + U(sampleIdx, 2), 0, -U(sampleIdx, 2) * X(sampleIdx, 0);
					C2.row(sampleIdx * 2 + 1) << 0, 0, -U(sampleIdx, 2) * X(sampleIdx, 1), U(sampleIdx, 2) * X(sampleIdx, 0), 0;

					C3.row(sampleIdx * 2) << X(sampleIdx, 2) * U(sampleIdx, 2);
					C3.row(sampleIdx * 2 + 1) << 0;
				}

				

				MatrixXd C11 = C1.transpose() * C1;
				MatrixXd C22 = C1.transpose() * C2;
				MatrixXd C33 = C1.transpose() * C3;

				MatrixXd M(9, 6);
				M << C33, C22;
				M = (-C11.fullPivLu().solve(M)).eval();

				MatrixXd K(6, 6);
				K.setZero();
				K(0, 3) = 1;

				K.block<3, 6>(1, 0) = M.block<3, 6>(0, 0);
				K.row(4) = M.row(5);
				K.row(5) = M.row(8);

				EigenSolver<MatrixXd> es(K);
				ArrayXcd D = es.eigenvalues();

				MatrixXcd sols(1, 6);
				sols.row(0) = 1. / D.transpose();

				// std::cout<< sols <<std::endl;

				for (size_t k = 0; k < 6; ++k) // std::numeric_limits<double>::epsilon()
				{
					if (abs(sols(0, k).imag()) > 0.01 || std::isnan(sols(0, k).real()) || std::isinf(sols(0, k).real()) || abs(sols(0, k).real()) > 10.0)
						continue;
					// Eigen::MatrixXd A(2*sample_number_, 9);
					MatrixXd A = MatrixXd::Zero(2 * sample_number_, 9);

					for (size_t kk = 0; kk < sample_number_; ++kk)
					{
						double z1 = 1 + sols(0, k).real() * X(kk, 2);
						double z2 = 1 + sols(0, k).real() * U(kk, 2);

						A.row(kk * 2) << 0, 0, 0, -z2 * X(kk, 0), -z2 * X(kk, 1), -z2 * z1, U(kk, 1) * X(kk, 0), U(kk, 1) * X(kk, 1), U(kk, 1) * z1;
						A.row(kk * 2 + 1) << z2 * X(kk, 0), z2 * X(kk, 1), z2 * z1, 0, 0, 0, -U(kk, 0) * X(kk, 0), -U(kk, 0) * X(kk, 1), -U(kk, 0) * z1;

					}

					// std::cout<< A <<std::endl;

					Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);

					const Eigen::Matrix<double, 9, 1> &null_space = svd.matrixV().col(8);

					if (null_space.hasNaN())
						continue;
					Eigen::Matrix3d H;
					H << null_space(0), null_space(1), null_space(2),
						null_space(3), null_space(4), null_space(5),
						null_space(6), null_space(7), null_space(8);

					// std::cout<< H <<std::endl;

					RadialHomography model;
					model.descriptor.block<3, 3>(0, 0) = H;
					model.descriptor(0, 3) = sols(0, k).real();
					model.descriptor(1, 3) = sols(0, k).real();
					models_.push_back(model);
				}

				if (!(models_.size() > 0))
				{
					RadialHomography model;
					model.descriptor.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
					model.descriptor(0, 3) = 0;
					model.descriptor(1, 3) = 0;
					models_.push_back(model);
				}

				return models_.size() > 0;
			} // namespace solver

		} // namespace solver
	}	  // namespace estimator
} // namespace gcransac
