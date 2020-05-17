/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  std::default_random_engine generator;
  std::normal_distribution<double> x_dist(x, std[0]);
  std::normal_distribution<double> y_dist(y, std[1]);
  std::normal_distribution<double> theta_dist(theta, std[2]);

  for (unsigned i = 0; i < kNumParticles; ++i) {
    Particle p;
    p.id = i;
    p.x = x_dist(generator);
    p.y = y_dist(generator);
    p.theta = theta_dist(generator);
    p.weight = 1.0;
    particles.push_back(p);
  }

}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
   for (auto & p : particles) {
     p.x = predictSingleAxis(p.x, p.theta, velocity, yaw_rate, delta_t, std_pos[0]);
     p.y = predictSingleAxis(p.y, p.theta, velocity, yaw_rate, delta_t, std_pos[1]);
     p.theta = predictTheta(p.theta, yaw_rate, delta_t, std_pos[2]);
   }

}

void ParticleFilter::dataAssociation(const vector<LandmarkObs>& predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */

   // predicted landmark measurements based on the map
   // observations are actual landmark measurements gathered from the lidar

   for ( auto & observation : observations) {
     double closest = std::numeric_limits<double>::max();
     int closest_id = 0;
     for (const auto & prediction : predicted) {
       double distance = dist(observation.x, observation.y, prediction.x, prediction.y);
       if (distance < closest) {
         closest = distance;
         closest_id = prediction.id;
       }
     }
     observation.id = closest_id;
   }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

   vector<LandmarkObs> predicted;
   for ( const auto & landmark : map_landmarks.landmark_list) {
     predicted.push_back(LandmarkObs(landmark));
   }

   double weightSum = 0.0;

   for (auto & p : particles) {
     std::vector<LandmarkObs> obsTransformed;
     for (const auto & observation : observations) {
       // For each observation, let's trasnform from vehicle to map coordinate system
       obsTransformed.push_back(transformObservation(p, observation));
     }
     // Now we have the observations in map frame from the point of view of our particle
     // Let's figure out which landmark in our map is closest to each observation
     dataAssociation(predicted, obsTransformed);

     // Now that we know which landmark is closest to each observation, let's multiply
     // all of the probabilities together to get our final probability
     double prob = 1.0;
     for (const auto & observation : obsTransformed) {
       LandmarkObs associatedLandmark(map_landmarks.landmark_list.at(observation.id-1));
       prob *= gaussian2D(observation.x, observation.y, associatedLandmark.x, associatedLandmark.y, 0.3, 0.3); //TODO sigma values
     }
     p.weight = prob;
     weightSum += p.weight;
   }

   // Normalize the weights
   for (auto & p : particles) {
     p.weight = p.weight / weightSum;
   }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

double ParticleFilter::predictSingleAxis(double pos0, double theta0, double velocity, double yaw_rate, double delta_t, double std) {
  std::default_random_engine generator;
  double pos = pos0 + (velocity / yaw_rate) * (::sin(theta0 + yaw_rate * delta_t) - ::sin(theta0));
  return std::normal_distribution<double>(pos, std)(generator);
}

double ParticleFilter::predictTheta(double theta0, double yaw_rate, double delta_t, double std) {
  std::default_random_engine generator;
  double theta = theta0 + yaw_rate * delta_t;
  return std::normal_distribution<double>(theta, std)(generator);
}

LandmarkObs ParticleFilter::transformObservation(const Particle & particle, const LandmarkObs & observation) {
  LandmarkObs result;
  result.id = observation.id;
  result.x = particle.x + (::cos(particle.theta) * observation.x) - (sin(particle.theta) * observation.y);
  result.y = particle.y + (::sin(particle.theta) * observation.x) + (cos(particle.theta) * observation.y);
  return result;
}

double ParticleFilter::gaussian2D(double x1, double y1, double x2, double y2, double sigmax, double sigmay) {
  double exponent = ::pow(x1 - x2, 2) / (2 * pow(sigmax, 2)) + ::pow(y1 - y2, 2) / (2 * pow(sigmay, 2));
  return (1.0 / (2 * M_PI * sigmax * sigmay)) * ::exp(-exponent);
}
