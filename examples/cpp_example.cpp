#include <vector>	
#include <mutex>	
#include <thread>
#include "utils.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Eigen> 
#include "opencv2/stitching.hpp" 
#include "opencv2/stitching/detail/autocalib.hpp" 
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>

#include "GCRANSAC.h"
#include "flann_neighborhood_graph.h"
#include "grid_neighborhood_graph.h"
#include "uniform_sampler.h"
#include "prosac_sampler.h"
#include "progressive_napsac_sampler.h"
#include "fundamental_estimator.h"
#include "homography_estimator.h"
#include "radial_homography_estimator.h"
#include "essential_estimator.h"
#include "rigid_transformation_estimator.h"
#include "preemption_sprt.h"
#include "preemption_empty.h"

#include "solver_stitching_gravity_equalfocal.h"
#include "solver_stitching_general_equalfocal.h"
#include "solver_stitching_gravity_varfocal.h"
#include "solver_stitching_general_varfocal.h"
#include "solver_homography_four_point.h"
#include "solver_stitching_general_ls_equal.h"

#include "solver_stitching_general_radial_equal_5pc.h"
#include "solver_radial_homography_5pc.h"
#include "solver_radial_homography_6pc.h"
#include "solver_stitching_general_radial_equal_3pt.h"
#include "solver_stitching_gravity_radial_equal.h"
#include "solver_stitching_gravity_radial_var.h"


#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>
#include <chrono>
// #include <experimental/filesystem>

#ifdef _WIN32
	#include <direct.h>
#endif 

#include <filesystem>
#include <set>
namespace fs = std::filesystem;
// namespace fs = std::experimental::filesystem;

struct stat info;

using namespace gcransac;

void detectCorrespondencesSIFT(
	const std::string &scene_,
	const size_t &sourceIndex_,
	const size_t &destinationIndex_,
	const cv::Mat &image1_,
	const cv::Mat &image2_,
	const std::string& correspondencePath_,
	cv::Mat &correspondences_); 

void visualizeStiching(
	const cv::Mat &image1_,
	const cv::Mat &image2_,
	const Eigen::Matrix3d &homography_);

template<class _Estimator, class ... _EstimatorTypes>
void processPhoneImagePair(
	const double& inlier_outlier_threshold_,
	const std::string& scene_,
	const size_t& frameSource_,
	const size_t& frameDestination_,
	const cv::Mat& image1_,
	const cv::Mat& image2_,
	const Eigen::Matrix3d& gravity1_,
	const Eigen::Matrix3d& gravity2_,
	const double& angleDifference_,
	const cv::Mat& correspondence_,
	const bool radialHomography_ = false,
	const bool visualizeMatches_ = false,
	const bool visualizeStiching_ = false);

std::pair<double, double> decomposeHomography(
	const Eigen::Matrix3d &homography_,
	const bool &varyingFocalLength_,
	Eigen::Matrix3d &rotation_);

typedef gcransac::estimator::RobustHomographyEstimator<
	gcransac::estimator::solver::HomographyFourPointSolver,
	gcransac::estimator::solver::HomographyFourPointSolver> Homography4PC;

typedef gcransac::estimator::RobustHomographyEstimator<
	gcransac::estimator::solver::StitchingGeneralEqual,
	gcransac::estimator::solver::HomographyFourPointSolver> HomographyEqual;

typedef gcransac::estimator::RobustHomographyEstimator<
	gcransac::estimator::solver::StitchingGravityEqual,
	gcransac::estimator::solver::HomographyFourPointSolver> HomographyGravityEqual;

typedef gcransac::estimator::RadialHomographyEstimator<
	gcransac::estimator::solver::StitchingGravityEqual,
	gcransac::estimator::solver::StitchingGeneralFocalEqualLS,1> HomographyGravityEqualLS;

typedef gcransac::estimator::RobustHomographyEstimator<
	gcransac::estimator::solver::StitchingGeneralVar,
	gcransac::estimator::solver::HomographyFourPointSolver> HomographyVar;

typedef gcransac::estimator::RobustHomographyEstimator<
	gcransac::estimator::solver::StitchingGravityVar,
	gcransac::estimator::solver::HomographyFourPointSolver> HomographyGravityVar;

typedef gcransac::estimator::RadialHomographyEstimator<
	gcransac::estimator::solver::StitchingGeneralRadialEqual5pc,
	gcransac::estimator::solver::RadialHomography6PC, 1> RadialHomographyEqual5PC;

typedef gcransac::estimator::RadialHomographyEstimator<
	gcransac::estimator::solver::RadialHomography5PC,
	gcransac::estimator::solver::RadialHomography6PC, 1> RadialHomography5PC;

typedef gcransac::estimator::RadialHomographyEstimator<
	gcransac::estimator::solver::RadialHomography6PC,
	gcransac::estimator::solver::RadialHomography6PC, 1> RadialHomography6PC;

typedef gcransac::estimator::RadialHomographyEstimator<
	gcransac::estimator::solver::StitchingGeneralRadialEqual3pt,
	gcransac::estimator::solver::RadialHomography6PC, 1> RadialHomography3PC;

typedef gcransac::estimator::RadialHomographyEstimator<
	gcransac::estimator::solver::StitchingGravityRadialVar,
	gcransac::estimator::solver::RadialHomography6PC, 1> RadialHomographyGravityVar;

typedef gcransac::estimator::RadialHomographyEstimator<
	gcransac::estimator::solver::StitchingGravityRadialEqual,
	gcransac::estimator::solver::RadialHomography6PC, 1> RadialHomographyGravityEqual;


void processPhoneData();

int main(int argc, const char* argv[])
{
	srand(static_cast<int>(time(NULL)));
	processPhoneData();
	return 0;
}


void processPhoneData()
{
	const std::string dataDirectory =
		"/home/deep/vodata/stitching/data/00/image_0/";
	constexpr double kInlierOutlierThreshold = 4.0;
	constexpr size_t kMinimumCorrespondenceNumber = 20;

	const std::string scene = "phone01";


	std::vector<cv::Mat> images;
	std::vector<std::string> imageNames;
	std::vector<Eigen::Matrix3d> gravities;
	std::set<fs::path> sorted_by_name;
	for (const auto& entry : fs::directory_iterator(dataDirectory))
	{
		if (entry.path().extension().string() != ".png")
			continue;

			sorted_by_name.insert(entry.path());

	}

	for (auto &filename : sorted_by_name)
    {
		const std::string imageName =
			filename.filename().string();

			std::cout << imageName << std::endl;
		imageNames.emplace_back(imageName);
		images.emplace_back(cv::imread(filename.string()));

		Eigen::Matrix3d gravity;
		std::ifstream gravityFile(dataDirectory + "frame_" + imageName.substr(0, imageName.size() - 4) + "_gravity.txt");
		for (size_t r = 0; r < 3; ++r)
			for (size_t c = 0; c < 3; ++c)
				gravityFile >> gravity(r, c);
		gravityFile.close();
		gravities.emplace_back(gravity);
	}

	std::vector<std::pair<size_t, size_t>> pairsToTest;
	pairsToTest.reserve(images.size() - 30);

	for (size_t sourceImageIdx = 0; sourceImageIdx < images.size() - 30; ++sourceImageIdx)
	{
		size_t destinationImageIdx = sourceImageIdx + 30;
			pairsToTest.emplace_back(std::make_pair(sourceImageIdx, destinationImageIdx));

			// std::cout << sourceImageIdx << " " << destinationImageIdx << std::endl;
	}

	// Iterating through the image pairs
#pragma omp parallel for num_threads(3)
	for (int pairIdx = 0; pairIdx < pairsToTest.size(); ++pairIdx) //pairsToTest.size()
	{
		// const auto& pair = pairsToTest[pairIdx];
		// const size_t& i = pair.first;
		// const size_t& j = pair.second;

		// const auto& pair = pairsToTest[pairIdx];
		const size_t& i = pairIdx;
		const size_t& j = pairIdx+30;
		const std::string correspondencePath =
			"/home/deep/vodata/stitching/data/00/correspondences/" + scene + "_" + imageNames[i].substr(0, imageNames[i].size() - 4) + "_" + 
			imageNames[j].substr(0, imageNames[j].size() - 4) + ".txt";

		printf("Processing view pair ('%s', '%s').\n",
			imageNames[i].c_str(), imageNames[j].c_str());
		printf("-----------------------------\n");

		// Load or detect SIFT correspondences
		cv::Mat correspondences;

		detectCorrespondencesSIFT(
			scene,
			i,
			j,
			images[i],
			images[j],
			correspondencePath,
			correspondences);

		printf("Correspondence number = %d\n", correspondences.rows);

		if (correspondences.rows < kMinimumCorrespondenceNumber)
		{
			printf("Not enough correspondences. Skipping the view pair.\n");
			continue;
		}

		// Testing arbitrary number of solvers by template packs
		processPhoneImagePair<
			Homography4PC,
			HomographyEqual,
			// HomographyGravityEqual,
			HomographyGravityEqualLS,
			HomographyVar,
			HomographyGravityVar,
			RadialHomographyGravityVar,
			RadialHomographyGravityEqual,
			RadialHomographyEqual5PC,
			RadialHomography5PC,
			RadialHomography6PC,
			RadialHomography3PC>(
				kInlierOutlierThreshold,
				scene,
				i,
				j,
				images[i],
				images[j],
				gravities[i],
				gravities[j],
				0,
				correspondences,
				true);
	}
}

double histogramVoting(
	const std::vector<std::pair<size_t, double>>& focalLengths_,
	const double &scale_,
	bool &success_)
{
	constexpr size_t kBinSize = 10;
	constexpr size_t kBinNumber = 200;
	std::vector<double> bins(kBinNumber, 0);
	std::vector<std::vector<std::pair<size_t, double>>> binToPoints(kBinNumber);
	size_t highestBin = 0;

	for (const auto& [inlierNumber, focalLength] : focalLengths_)
	{
		const double scaleFocal =
			(1.0 - focalLength) * scale_;
		const double discreteFocal =
			round(scaleFocal / kBinSize);

		if (discreteFocal < 0 ||
			discreteFocal >= kBinNumber)
			continue;

		bins[discreteFocal] += inlierNumber;
		binToPoints[discreteFocal].emplace_back(std::make_pair(inlierNumber, focalLength));
		if (bins[discreteFocal] > bins[highestBin])
			highestBin = discreteFocal;
	}

	success_ = false;
	double averageFocalLength = 0;
	double weightSum = 0;
	for (const auto& [inlierNumber, focalLength] : binToPoints[highestBin])
	{
		averageFocalLength += inlierNumber * focalLength;
		weightSum += inlierNumber;
		success_ = true;
	}

	averageFocalLength = (averageFocalLength / weightSum) * scale_;

	return averageFocalLength;
}

static std::mutex savingMutex;

// Parameter pack so it can be called with multiple estimators to be tested
// and it automatically tests them.
template<class _Estimator, class ... _EstimatorTypes>
void processPhoneImagePair(
	const double& inlier_outlier_threshold_,
	const std::string& scene_,
	const size_t& frameSource_,
	const size_t& frameDestination_,
	const cv::Mat& image1_,
	const cv::Mat& image2_,
	const Eigen::Matrix3d& gravity1_,
	const Eigen::Matrix3d& gravity2_,
	const double& angleDifference_,
	const cv::Mat& correspondences_,
	const bool radialHomography_,
	const bool visualizeMatches_,
	const bool visualizeStiching_)
{
	printf("-----------------------------\n");
	// The ground truth focal length for the current scene
	constexpr double kGroundTruthFocalLength = 480.0;
	Eigen::Matrix3d gtK1, gtK2;
	gtK1 << kGroundTruthFocalLength, 0, image1_.cols / 2.0,
		0, kGroundTruthFocalLength, image1_.rows / 2.0,
		0, 0, 1;
	gtK2 << kGroundTruthFocalLength, 0, image2_.cols / 2.0,
		0, kGroundTruthFocalLength, image2_.rows / 2.0,
		0, 0, 1;

	// Normalize the correspondences by the image size
	Eigen::Matrix3d K2, K1;

	// double MAX(image1_.cols, image1_.rows) = 1088.0; // MAX(image1_.cols, image1_.rows)

	K1 << MAX(image1_.cols, image1_.rows), 0, image1_.cols / 2.0,
		0, MAX(image1_.cols, image1_.rows), image1_.rows / 2.0,
		0, 0, 1;
	K2 << MAX(image1_.cols, image1_.rows), 0, image2_.cols / 2.0,
		0, MAX(image1_.cols, image1_.rows), image2_.rows / 2.0,
		0, 0, 1;

	Eigen::Matrix3d K1inv = K1.inverse(),
		K2inv = K2.inverse();

	Eigen::Matrix3d translation1,
		translation2;
	translation1 <<
		1, 0, image1_.cols / 2.0,
		0, 1, image1_.rows / 2.0,
		0, 0, 1;
	translation2 <<
		1, 0, image2_.cols / 2.0,
		0, 1, image2_.rows / 2.0,
		0, 0, 1;
	Eigen::Matrix3d scaling1,
		scaling2;
	scaling1 <<
		MAX(image1_.cols, image1_.rows), 0, 0,
		0, MAX(image1_.cols, image1_.rows), 0,
		0, 0, 1;
	scaling2 <<
		MAX(image1_.cols, image1_.rows), 0, 0,
		0, MAX(image1_.cols, image1_.rows), 0,
		0, 0, 1;

	_Estimator estimator(K1, K2, &correspondences_);
	Model model;

	double normalizedThreshold = inlier_outlier_threshold_ / MAX(image1_.cols, image1_.rows);

	printf("Minimal solver = %s\nNon-minimal solver = %s\n",
		estimator.getMinimalSolver().getName(),
		estimator.getNonMinimalSolver().getName());

	cv::Mat normalizedCorrespondences(correspondences_.rows, 2 * correspondences_.cols, correspondences_.type());
	for (size_t i = 0; i < correspondences_.rows; ++i)
	{
		Eigen::Vector3d pt1, pt2;
		pt1 << correspondences_.at<double>(i, 0), correspondences_.at<double>(i, 1), 1;
		pt2 << correspondences_.at<double>(i, 2), correspondences_.at<double>(i, 3), 1;
		pt1 = K1inv * pt1;
		pt2 = K2inv * pt2;

		normalizedCorrespondences.at<double>(i, 0) = pt1(0);
		normalizedCorrespondences.at<double>(i, 1) = pt1(1);
		normalizedCorrespondences.at<double>(i, 2) = pt2(0);
		normalizedCorrespondences.at<double>(i, 3) = pt2(1);
		normalizedCorrespondences.at<double>(i, 4) = pt1(0);
		normalizedCorrespondences.at<double>(i, 5) = pt1(1);
		normalizedCorrespondences.at<double>(i, 6) = pt2(0);
		normalizedCorrespondences.at<double>(i, 7) = pt2(1);
	}

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	neighborhood::FlannNeighborhoodGraph neighborhood(&correspondences_, 20.0);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("Neighborhood calculation time = %f secs\n", elapsed_seconds.count());

	if (estimator.getMinimalSolver().needsGravity())
		estimator.getMutableMinimalSolver().setGravity(gravity1_, gravity2_);

	preemption::SPRTPreemptiveVerfication<_Estimator> sprt(normalizedCorrespondences,
		estimator);
	preemption::EmptyPreemptiveVerfication<_Estimator> emptyPreemption;

	GCRANSAC<_Estimator,
		neighborhood::FlannNeighborhoodGraph,
		MSACScoringFunction<_Estimator>,
		preemption::SPRTPreemptiveVerfication<_Estimator>> gcransac;
	gcransac.settings.threshold = normalizedThreshold; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = 0.95; // The weight of the spatial coherence term
	gcransac.settings.confidence = 0.99; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
	gcransac.settings.max_iteration_number = 5000; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 100; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = 20; // The radius of the neighborhood ball

	sampler::UniformSampler main_sampler(&normalizedCorrespondences);// , estimator.sampleSize()); // The main sampler is used inside the local optimization
	sampler::UniformSampler local_optimization_sampler(&normalizedCorrespondences); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return;
	}

	// Start GC-RANSAC
	gcransac.run(normalizedCorrespondences,
		estimator,
		&main_sampler,
		&local_optimization_sampler,
		&neighborhood,
		model,
		sprt);

	// Get the statistics of the results
	const utils::RANSACStatistics& statistics_ =
		gcransac.getRansacStatistics();

	// Get the focal length by histogram voting
	bool histogramVotingSuccess = false;
	double votedFocalLength = 0;
	if (gcransac.getFocalLengths().size() > 0)
		votedFocalLength = histogramVoting(gcransac.getFocalLengths(), MAX(image1_.cols, image1_.rows), histogramVotingSuccess);

	const size_t& inlierNumber = statistics_.inliers.size();
	 
	printf("Inlier number = %d\nProcessing time = %f\nIteration number = %d\n",
		inlierNumber,
		statistics_.processing_time,
		statistics_.iteration_number);

	const double angle = angleDifference_ / 180.0 * M_PI;

	// Decompose the homography
	savingMutex.lock();
	if (model.descriptor.rows() == 3 && statistics_.inliers.size() >= 4)
	{
		std::vector<gcransac::Model> models;

		model.descriptor.block<3, 3>(0, 0) = 
			scaling2 * model.descriptor.block<3, 3>(0, 0) * scaling1.inverse();
		Eigen::Matrix3d decomposableHomography = model.descriptor.block<3, 3>(0, 0);
		decomposableHomography = decomposableHomography / decomposableHomography(1, 1);

		Eigen::Matrix3d rotation;
		std::pair<double, double> focalLengthsEstimated =
			decomposeHomography(scaling2.inverse() * model.descriptor.block<3, 3>(0, 0) * scaling1,
				true,
				rotation);
		focalLengthsEstimated.first = focalLengthsEstimated.first * MAX(image1_.cols, image1_.rows);
		focalLengthsEstimated.second = focalLengthsEstimated.second * MAX(image1_.cols, image1_.rows);

		std::cout << "1. focal length from decomposition = " << focalLengthsEstimated.first << std::endl;
		std::cout << "2. focal length from decomposition = " << focalLengthsEstimated.second << std::endl;
		std::cout << "Voted focal length = " << votedFocalLength << std::endl;

		double focalLengthError =
			abs(sqrt(focalLengthsEstimated.first * focalLengthsEstimated.second) - kGroundTruthFocalLength)/kGroundTruthFocalLength;
		double votedFocalLengthError =
			abs(votedFocalLength - kGroundTruthFocalLength)/kGroundTruthFocalLength;

		if (isnan(focalLengthError))
			focalLengthError = 1;
		if (isnan(votedFocalLengthError))
			votedFocalLengthError = 1;

		std::cout << "Focal length error = " << focalLengthError << std::endl;
		std::cout << "Voted focal length error = " << votedFocalLengthError << std::endl;

		std::ofstream resultFile("phone_results.csv", std::fstream::app);

		resultFile << scene_ << ";"
			<< frameSource_ << ";"
			<< frameDestination_ << ";"
			<< angle << ";"
			<< estimator.getMinimalSolver().getName() << ";"
			<< estimator.getNonMinimalSolver().getName() << ";"
			<< inlierNumber << ";"
			<< statistics_.processing_time << ";"
			<< statistics_.iteration_number << ";"
			<< focalLengthError << ";"
			<< votedFocalLengthError << std::endl;
		resultFile.close();
	}
	else
	{
		std::ofstream resultFile("phone_results.csv", std::fstream::app);
		resultFile << scene_ << ";"
			<< frameSource_ << ";"
			<< frameDestination_ << ";"
			<< angle << ";"
			<< estimator.getMinimalSolver().getName() << ";"
			<< estimator.getNonMinimalSolver().getName() << ";"
			<< inlierNumber << ";"
			<< statistics_.processing_time << ";"
			<< statistics_.iteration_number << ";"
			<< -1 << ";"
			<< -1 << std::endl;
		resultFile.close();
	}
	savingMutex.unlock();

	if (inlierNumber < 20)
	{
		printf("Too few inliers are found (%d < %d).\n", inlierNumber, 20);
	}
	else
	{
		if (visualizeMatches_)
		{
			std::vector<cv::KeyPoint> keypoints1,
				keypoints2;
			std::vector<cv::DMatch> matches;
			keypoints1.reserve(inlierNumber);
			keypoints2.reserve(inlierNumber);
			matches.reserve(inlierNumber);

			size_t index = 0;
			for (const auto& inlierIdx : statistics_.inliers)
			{
				keypoints1.emplace_back(cv::KeyPoint(correspondences_.at<double>(inlierIdx, 0), correspondences_.at<double>(inlierIdx, 1), 0));
				keypoints2.emplace_back(cv::KeyPoint(correspondences_.at<double>(inlierIdx, 2), correspondences_.at<double>(inlierIdx, 3), 0));
				matches.emplace_back(cv::DMatch(index, index, 0));
				++index;
			}

			cv::Mat outputImage;

			cv::drawMatches(image1_,
				keypoints1,
				image2_,
				keypoints2,
				matches,
				outputImage);

			cv::imshow("Image", outputImage);
			cv::waitKey(0);
		}

		if (visualizeStiching_)
		{
			visualizeStiching(
				image1_,
				image2_,
				translation2 * model.descriptor.block<3, 3>(0, 0) * translation1.inverse());
		}
	}

	// If there is an untested estimator run the function again
	if constexpr (sizeof...(_EstimatorTypes) > 0) {
		processPhoneImagePair<_EstimatorTypes...>(
			inlier_outlier_threshold_,
			scene_,
			frameSource_,
			frameDestination_,
			image1_,
			image2_,
			gravity1_,
			gravity2_,
			angleDifference_,
			correspondences_,
			radialHomography_,
			visualizeMatches_,
			visualizeStiching_);
	}
}

void visualizeStiching(
	const cv::Mat &image1_,
	const cv::Mat &image2_,
	const Eigen::Matrix3d &homography_)
{
	cv::Mat warpedImage;
	cv::Mat homography = (cv::Mat_<double>(3, 3) <<
		homography_(0, 0), homography_(0, 1), homography_(0, 2),
		homography_(1, 0), homography_(1, 1), homography_(1, 2),
		homography_(2, 0), homography_(2, 1), homography_(2, 2));

	cv::Mat inverseHomography = homography.inv();

	// Calculating the dimensions of the output image
	cv::Mat pt1, pt2, pt3, pt4;
	pt1 = (cv::Mat_<double>(3, 1) << 0, 0, 1);
	pt2 = (cv::Mat_<double>(3, 1) << image2_.cols, 0, 1);
	pt3 = (cv::Mat_<double>(3, 1) << image2_.cols, image2_.rows, 1);
	pt4 = (cv::Mat_<double>(3, 1) << 0, image2_.rows, 1);

	pt1 = inverseHomography * pt1;
	pt2 = inverseHomography * pt2;
	pt3 = inverseHomography * pt3;
	pt4 = inverseHomography * pt4;

	pt1 = pt1 / pt1.at<double>(2);
	pt2 = pt2 / pt2.at<double>(2);
	pt3 = pt3 / pt3.at<double>(2);
	pt4 = pt4 / pt4.at<double>(2);

	double minX = MIN(pt1.at<double>(0), MIN(pt2.at<double>(0), MIN(pt3.at<double>(0), pt4.at<double>(0))));
	double maxX = MAX(pt1.at<double>(0), MAX(pt2.at<double>(0), MAX(pt3.at<double>(0), pt4.at<double>(0))));
	double minY = MIN(pt1.at<double>(1), MIN(pt2.at<double>(1), MIN(pt3.at<double>(1), pt4.at<double>(1))));
	double maxY = MAX(pt1.at<double>(1), MAX(pt2.at<double>(1), MAX(pt3.at<double>(1), pt4.at<double>(1))));

	double width = maxX - minX;
	double height = maxY - minY;

	if (width > 10000 || height > 10000)
	{
		printf("The warped image is too big to be shown.\n");
		return;
	}

	cv::Mat translation = cv::Mat::eye(3, 3, CV_64F);
	translation.at<double>(0, 2) = MAX(0, -minX);
	translation.at<double>(1, 2) = MAX(0, -minY);
	inverseHomography = translation * inverseHomography;

	cv::warpPerspective(
		image2_,
		warpedImage,
		inverseHomography,
		cv::Size(MAX(image1_.cols, width + translation.at<double>(0, 2)) + 1,
			MAX(image1_.rows, height + translation.at<double>(1, 2)) + 1));

	cv::Mat roi1(warpedImage, cv::Rect(translation.at<double>(0, 2), translation.at<double>(1, 2), image1_.cols, image1_.rows));
	image1_.copyTo(roi1);

	cv::namedWindow("Warped", cv::WINDOW_NORMAL);
	cv::imshow("Warped", warpedImage);
	cv::waitKey(0);
}

void detectCorrespondencesSIFT(
	const std::string &scene_,
	const size_t &sourceIndex_,
	const size_t &destinationIndex_,
	const cv::Mat &image1_,
	const cv::Mat &image2_,
	const std::string &correspondencePath_,
	cv::Mat &correspondences_) 
{
	std::ifstream file(correspondencePath_);

	if (file.is_open())
	{
		// First row is the correspondence number
		size_t n;
		file >> n;

		correspondences_.create(n, 4, CV_64F);
		double *correspondences_ptr = reinterpret_cast<double *>(correspondences_.data);
		while (file >> *(correspondences_ptr++));
		file.close();
		return;
	}

	std::vector<std::vector<cv::KeyPoint>> keypoints(2);
	std::vector<cv::Mat> descriptors(2);
	// cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> sift = cv::xfeatures2d::SIFT::create(8000, 3, 0.0, 10000.);
	auto sift = cv::SIFT::create(8000, 3, 0.0, 10000.);

	int widths[2], heights[2];
	cv::Mat temp;
	std::vector<const cv::Mat *> images = { &image1_, &image2_};
	for (auto imageIndex = 0; imageIndex < 2; ++imageIndex) 
	{
		widths[imageIndex] = images[imageIndex]->cols;
		heights[imageIndex] = images[imageIndex]->rows;

		sift->detectAndCompute(*images[imageIndex], cv::noArray(), keypoints[imageIndex], descriptors[imageIndex]);
		for (int i = 0; i <= descriptors[imageIndex].rows - 1; i++)
		{
			cv::normalize(descriptors[imageIndex].row(i), temp, 1.0, 0, cv::NORM_L1);
			cv::sqrt(temp.clone(), temp);
			temp.copyTo(descriptors[imageIndex].rowRange(i, i+1));
		}

		printf("Detected SIFT keypoint number on the %d-th image = %d\n", imageIndex + 1, keypoints[imageIndex].size());

		if (keypoints[imageIndex].size() == 0)
			return;
	}

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	std::vector<std::vector<cv::DMatch>> matches;
	matcher->knnMatch(descriptors[0], descriptors[1], matches, 2);

	cv::Ptr<cv::DescriptorMatcher> matcher_opposite = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	std::vector<std::vector<cv::DMatch>> matches_opposite;
	matcher_opposite->knnMatch(descriptors[1], descriptors[0], matches_opposite, 1);

	std::vector<std::pair<double, const std::vector<cv::DMatch> *>> good_matches;

	for (size_t i = 0; i < matches.size(); ++i)
		if ((matches[i][0].distance < 0.85 * matches[i][1].distance) &&
			(matches[i][0].queryIdx == matches_opposite[matches[i][0].trainIdx][0].trainIdx)) // We increased threshold for mutual snn check
			good_matches.emplace_back(std::make_pair(matches[i][0].distance / matches[i][1].distance, &matches[i]));

	// Sort for PROSAC
	std::sort(good_matches.begin(), good_matches.end());

	correspondences_.create(good_matches.size(), 4, CV_64F);
	double *correspondences_ptr = reinterpret_cast<double *>(correspondences_.data);

	std::ofstream outFile(correspondencePath_);
	outFile << good_matches.size() << "\n";

	for (const auto &match_ptr : good_matches)
	{
		const std::vector<cv::DMatch> &match = *match_ptr.second;

		const double &x1 = keypoints[0][match[0].queryIdx].pt.x,
			&y1 = keypoints[0][match[0].queryIdx].pt.y,
			&x2 = keypoints[1][match[0].trainIdx].pt.x,
			&y2 = keypoints[1][match[0].trainIdx].pt.y;

		*(correspondences_ptr++) = x1;
		*(correspondences_ptr++) = y1;
		*(correspondences_ptr++) = x2;
		*(correspondences_ptr++) = y2;

		outFile << x1 << " " << y1 << " " << x2 << " " << y2 << "\n";
	}

	outFile.close();
}

std::pair<double, double> decomposeHomography(
	const Eigen::Matrix3d &homography_,
	const bool &varyingFocalLength_,
	Eigen::Matrix3d &rotation_)
{
	Eigen::Matrix<double, 3, 2> AA;

	AA << homography_(0, 0) * homography_(1, 0) + homography_(0, 1) * homography_(1, 1), homography_(0, 2) * homography_(1, 2),
		homography_(0, 0) * homography_(2, 0) + homography_(0, 1) * homography_(2, 1), homography_(0, 2)* homography_(2, 2),
		homography_(1, 0) * homography_(2, 0) + homography_(1, 1) * homography_(2, 1), homography_(1, 2)* homography_(2, 2);
	
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(AA, Eigen::ComputeFullV);
	const Eigen::Matrix<double, 2, 1> &V2a = svd.matrixV().col(1);
	double fa = sqrt(V2a(0) / V2a(1));

	Eigen::Matrix<double, 3, 3> inverseHomography = homography_.inverse();
	
	AA << inverseHomography(0, 0) * inverseHomography(1, 0) + inverseHomography(0, 1) * inverseHomography(1, 1), inverseHomography(0, 2) * inverseHomography(1, 2),
		inverseHomography(0, 0) * inverseHomography(2, 0) + inverseHomography(0, 1) * inverseHomography(2, 1), inverseHomography(0, 2)*inverseHomography(2, 2),
		inverseHomography(1, 0) * inverseHomography(2, 0) + inverseHomography(1, 1) * inverseHomography(2, 1), inverseHomography(1, 2)*inverseHomography(2, 2);
	Eigen::JacobiSVD<Eigen::MatrixXd> svd2(AA, Eigen::ComputeFullV);
	const Eigen::Matrix<double, 2, 1> &V2b = svd2.matrixV().col(1);
	double fb = sqrt(V2b(0) / V2b(1));

	if (isnan(fa))
		fa = fb;
	if (isnan(fb))
		fb = fa;

	double f = sqrt(fa * fb); // if equal focal 
	
	Eigen::Matrix3d K2, K1;
	K1 << f, 0, 0,
		0, f, 0,
		0, 0, 1;
	K2 << f, 0, 0,
		0, f, 0,
		0, 0, 1;

	Eigen::JacobiSVD<Eigen::MatrixXd> svd3(K2.inverse() * homography_ * K1, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3d U = svd3.matrixU();
	Eigen::Matrix3d V = svd3.matrixV().transpose();

	Eigen::Matrix3d S;
	S << 1, 0, 0,
		0, 1, 0,
		0, 0, ((U * V).determinant() > 0) ? 1 : -1;

	rotation_ = U * S * V;
	
	return std::make_pair(fa, fb);
}