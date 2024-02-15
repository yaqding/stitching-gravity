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
			class StitchingGravityVar : public SolverEngine
			{
			public:
				// StitchingGravityVar() : gravity_source(Eigen::Matrix3d::Identity()),
				// 						gravity_destination(Eigen::Matrix3d::Identity())
				// {
				// }

				~StitchingGravityVar()
				{
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 2;
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return true;
				}

				static constexpr size_t maximumSolutions()
				{
					return 4;
				}

				static constexpr char *getName()
				{
					return "Gravity Var Focal (2PC)";
				}

				static constexpr bool needsGravity()
				{
					return true;
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

				inline double sign2(const std::complex<double> z) const;
				void solve_quartic(double b, double c, double d, double e, std::complex<double> roots[4]) const;
			};

			double StitchingGravityVar::sign2(const std::complex<double> z) const
			{
				if (std::abs(z.real()) > std::abs(z.imag()))
					return z.real() < 0 ? -1.0 : 1.0;
				else
					return z.imag() < 0 ? -1.0 : 1.0;
			}

			void StitchingGravityVar::solve_quartic(double b, double c, double d, double e, std::complex<double> roots[4]) const
			{

				// Find depressed quartic
				std::complex<double> p = c - 3.0 * b * b / 8.0;
				std::complex<double> q = b * b * b / 8.0 - 0.5 * b * c + d;
				std::complex<double> r = (-3.0 * b * b * b * b + 256.0 * e - 64.0 * b * d + 16.0 * b * b * c) / 256.0;

				// Resolvent cubic is now
				// U^3 + 2*p U^2 + (p^2 - 4*r) * U - q^2
				std::complex<double> bb = 2.0 * p;
				std::complex<double> cc = p * p - 4.0 * r;
				std::complex<double> dd = -q * q;

				// Solve resolvent cubic
				std::complex<double> d0 = bb * bb - 3.0 * cc;
				std::complex<double> d1 = 2.0 * bb * bb * bb - 9.0 * bb * cc + 27.0 * dd;

				std::complex<double> C3 = (d1.real() < 0) ? (d1 - sqrt(d1 * d1 - 4.0 * d0 * d0 * d0)) / 2.0 : (d1 + sqrt(d1 * d1 - 4.0 * d0 * d0 * d0)) / 2.0;

				std::complex<double> C;
				if (C3.real() < 0)
					C = -std::pow(-C3, 1.0 / 3);
				else
					C = std::pow(C3, 1.0 / 3);

				std::complex<double> u2 = (bb + C + d0 / C) / -3.0;

				//std::complex<double> db = u2 * u2 * u2 + bb * u2 * u2 + cc * u2 + dd;

				std::complex<double> u = sqrt(u2);

				std::complex<double> s = -u;
				std::complex<double> t = (p + u * u + q / u) / 2.0;
				std::complex<double> v = (p + u * u - q / u) / 2.0;

				roots[0] = (-u - sign2(u) * sqrt(u * u - 4.0 * v)) / 2.0;
				roots[1] = v / roots[0];
				roots[2] = (-s - sign2(s) * sqrt(s * s - 4.0 * t)) / 2.0;
				roots[3] = t / roots[2];

				for (int i = 0; i < 4; i++)
				{
					roots[i] = roots[i] - b / 4.0;

					// do one step of newton refinement
					std::complex<double> x = roots[i];
					std::complex<double> x2 = x * x;
					std::complex<double> x3 = x * x2;
					std::complex<double> dx = -(x2 * x2 + b * x3 + c * x2 + d * x + e) / (4.0 * x3 + 3.0 * b * x2 + 2.0 * c * x + d);
					roots[i] = x + dx;
				}
			}

			OLGA_INLINE bool StitchingGravityVar::estimateModel(
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

				// const size_t &sampleIdx = sample_[0];

				const double
					&u1 = data_.at<double>(sample_[0], 0),
					&v1 = data_.at<double>(sample_[0], 1),
					&u2 = data_.at<double>(sample_[0], 2),
					&v2 = data_.at<double>(sample_[0], 3),
					&u3 = data_.at<double>(sample_[1], 0),
					&v3 = data_.at<double>(sample_[1], 1),
					&u4 = data_.at<double>(sample_[1], 2),
					&v4 = data_.at<double>(sample_[1], 3);

				if (gravity_source(0, 0) != 1)
				{

					double a1 = gravity_source(0, 1) / gravity_source(0, 0);
					double a2 = gravity_source(1, 0) / gravity_source(0, 0);
					double a3 = gravity_source(1, 1) / gravity_source(0, 0);
					double a4 = gravity_source(1, 2) / gravity_source(0, 0);
					double a5 = gravity_source(2, 0) / gravity_source(0, 0);
					double a6 = gravity_source(2, 1) / gravity_source(0, 0);
					double a7 = gravity_source(2, 2) / gravity_source(0, 0);

					double b1 = gravity_destination(0, 1) / gravity_destination(0, 0);
					double b2 = gravity_destination(1, 0) / gravity_destination(0, 0);
					double b3 = gravity_destination(1, 1) / gravity_destination(0, 0);
					double b4 = gravity_destination(1, 2) / gravity_destination(0, 0);
					double b5 = gravity_destination(2, 0) / gravity_destination(0, 0);
					double b6 = gravity_destination(2, 1) / gravity_destination(0, 0);
					double b7 = gravity_destination(2, 2) / gravity_destination(0, 0);

					double d1 = u1 + a1 * v1;
					double d2 = a2 * u1 + a3 * v1;
					double d3 = a5 * u1 + a6 * v1;

					double d4 = u3 + a1 * v3;
					double d5 = a2 * u3 + a3 * v3;
					double d6 = a5 * u3 + a6 * v3;

					double e1 = d1 + b2 * d2 + b5 * d3;
					double e2 = 2 * d3 - 2 * b5 * d1;
					double e3 = a4 * b2 + a7 * b5;
					double e4 = 2 * a7;
					double e5 = -d1 + b2 * d2 - b5 * d3;
					double e6 = a4 * b2 - a7 * b5;

					double e7 = d4 + b2 * d5 + b5 * d6;
					double e8 = 2 * d6 - 2 * b5 * d4;
					double e9 = a4 * b2 + a7 * b5;
					double e10 = 2 * a7;
					double e11 = -d4 + b2 * d5 - b5 * d6;
					double e12 = a4 * b2 - a7 * b5;

					double f1 = b1 * d1 + b3 * d2 + b6 * d3;
					double f2 = 2 * b1 * d3 - 2 * b6 * d1;
					double f3 = a4 * b3 + a7 * b6;
					double f4 = 2 * a7 * b1;
					double f5 = -b1 * d1 + b3 * d2 - b6 * d3;
					double f6 = a4 * b3 - a7 * b6;

					double f7 = b1 * d4 + b3 * d5 + b6 * d6;
					double f8 = 2 * b1 * d6 - 2 * b6 * d4;
					double f9 = a4 * b3 + a7 * b6;
					double f10 = 2 * a7 * b1;
					double f11 = -b1 * d4 + b3 * d5 - b6 * d6;
					double f12 = a4 * b3 - a7 * b6;

					double g1 = b4 * d2 + b7 * d3;
					double g2 = -2 * b7 * d1;
					double g3 = a4 * b4 + a7 * b7;
					double g4 = b4 * d2 - b7 * d3;
					double g5 = a4 * b4 - a7 * b7;

					double n0 = f1 * u2 - e1 * v2;
					double n1 = f2 * u2 - e2 * v2;
					double n2 = f5 * u2 - e5 * v2;
					double n3 = f3 * u2 - e3 * v2;
					double n4 = f4 * u2 - e4 * v2;
					double n5 = f6 * u2 - e6 * v2;

					double n6 = f7 * u4 - e7 * v4;
					double n7 = f8 * u4 - e8 * v4;
					double n8 = f11 * u4 - e11 * v4;
					double n9 = f9 * u4 - e9 * v4;
					double n10 = f10 * u4 - e10 * v4;
					double n11 = f12 * u4 - e12 * v4;

					double c4 = n2 * n11 - n5 * n8;
					double c3 = n1 * n11 + n2 * n10 - n4 * n8 - n5 * n7;
					double c2 = n0 * n11 + n1 * n10 + n2 * n9 - n3 * n8 - n4 * n7 - n5 * n6;
					double c1 = n0 * n10 + n1 * n9 - n3 * n7 - n4 * n6;
					double c0 = n0 * n9 - n3 * n6;

					c0 /= c4;
					c1 /= c4;
					c2 /= c4;
					c3 /= c4;

					std::complex<double> roots[4];
					solve_quartic(c3, c2, c1, c0, roots);

					for (size_t k = 0; k < 4; ++k)
					{
						if (roots[k].imag() > std::numeric_limits<double>::epsilon() || roots[k].real() > 1 || roots[k].real() < -1)
							continue;

						double r = roots[k].real();
						double rsqr = r * r;
						double fa = -(n0 + n1 * r + n2 * rsqr) / (n3 + n4 * r + n5 * rsqr);
						double fb = v2 * (g1 + r * g2 + fa * g3 + rsqr * g4 + rsqr * fa * g5) / (f1 + r * f2 + fa * f3 + r * fa * f4 + rsqr * f5 + rsqr * fa * f6);

						if (fa < 0 || fb < 0)
							continue;

						Eigen::Matrix<double, 3, 3> Ry;
						Ry << (1 - rsqr), 0, 2 * r,
							0, 1 + rsqr, 0,
							-2 * r, 0, (1 - rsqr);

						Eigen::Matrix<double, 3, 3> K2;
						K2 << fb, 0, 0,
							0, fb, 0,
							0, 0, 1.0;

						Eigen::Matrix<double, 3, 3> K1inv;
						K1inv << 1.0 / fa, 0, 0,
							0, 1.0 / fa, 0,
							0, 0, 1.0;

						Eigen::Matrix<double, 3, 3> H;
						H = K2 * gravity_destination.transpose() * Ry * gravity_source * K1inv;

						Homography model;
						model.descriptor = Eigen::MatrixXd(3, 4);
						model.descriptor.block<3, 3>(0, 0) = H;
						model.descriptor(2, 3) = sqrt(fa * fb);
						models_.push_back(model);
					}
				}
				else
				{
					double c0 = u2 * v1 * v4 - u1 * v2 * v4 + u3 * v2 * v4 - u4 * v2 * v3;
					double c2 = u1 * v2 * v4 + u2 * v1 * v4 - u3 * v2 * v4 - u4 * v2 * v3;

					// std::cout << c0 << std::endl;
					// std::cout << c2 << std::endl;

					double rsqr = -c0/c2;
					std::complex<double> roots[2];
					roots[0] = sqrt(rsqr);
					roots[1] = -sqrt(rsqr);
					// std::cout << roots[0] << std::endl;
					// std::cout << roots[1] << std::endl;

					for (size_t k = 0; k < 2; ++k)
					{
						if (roots[k].imag() > std::numeric_limits<double>::epsilon() || roots[k].real() > 1 || roots[k].real() < -1) // cos(theta) 1-s^2 > 0
							continue;

						double r = roots[k].real();
						// double rsqr = r * r;
						double fa = (u2 * v1 - u1 * v2 + rsqr * u1 * v2 + rsqr * u2 * v1) / (2 * r * v2);
						double fb = (fa*v2  - fa*rsqr*v2  - 2*r*u1*v2)/(v1 + rsqr*v1);

						if (fa < 0 || fb < 0)
							continue;

						Eigen::Matrix<double, 3, 3> Ry;
						Ry << (1 - rsqr), 0, 2 * r,
							0, 1 + rsqr, 0,
							-2 * r, 0, (1 - rsqr);

						Eigen::Matrix<double, 3, 3> K2;
						K2 << fb, 0, 0,
							0, fb, 0,
							0, 0, 1.0;

						Eigen::Matrix<double, 3, 3> K1inv;
						K1inv << 1.0 / fa, 0, 0,
							0, 1.0 / fa, 0,
							0, 0, 1.0;

						Eigen::Matrix<double, 3, 3> H;
						H = K2 * gravity_destination.transpose() * Ry * gravity_source * K1inv;

						Homography model;
						model.descriptor = Eigen::MatrixXd(3, 4);
						model.descriptor.block<3, 3>(0, 0) = H;
						model.descriptor(2, 3) = sqrt(fa * fb);
						models_.push_back(model);
					}
				}

				return models_.size() > 0;
			}

		} // namespace solver
	}	  // namespace estimator
} // namespace gcransac
