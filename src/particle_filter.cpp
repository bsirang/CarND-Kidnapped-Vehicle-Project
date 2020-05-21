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
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

std::ostream& operator<<(std::ostream& os, const Particle& p) {
  return os << "[x = " << p.x << " y = " << p.y << " theta = " << p.theta << " weight = " << p.weight << "]";
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  std::normal_distribution<double> x_dist(x, std[0]);
  std::normal_distribution<double> y_dist(y, std[1]);
  std::normal_distribution<double> theta_dist(theta, std[2]);

  if (debug) std::cout << "Initialization" << std::endl;
  for (unsigned i = 0; i < kNumParticles; ++i) {
    Particle p;
    p.id = i;
    p.x = x_dist(generator);
    p.y = y_dist(generator);
    p.theta = theta_dist(generator);
    p.weight = 1.0;
    if (debug) std::cout << p << std::endl;
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
   if (debug) std::cout << "Prediction with vel" << velocity << " and yaw_rate " << yaw_rate << std::endl;
   for (auto & p : particles) {
     Particle old = p;
     p.x = predictXAxis(p.x, p.theta, velocity, yaw_rate, delta_t, std_pos[0]);
     p.y = predictYAxis(p.y, p.theta, velocity, yaw_rate, delta_t, std_pos[1]);
     p.theta = predictTheta(p.theta, yaw_rate, delta_t, std_pos[2]);
     if (debug) std::cout << old << " => " << p << std::endl;
   }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {

   double weightSum = 0.0;
   if (debug) std::cout << "Update Weights" << std::endl;
   for (auto & p : particles) {
     std::vector<int> associations;
     std::vector<double> sense_x;
     std::vector<double> sense_y;

     std::vector<LandmarkObs> obsTransformed;
     for (const auto & observation : observations) {
       // For each observation, let's trasnform from vehicle to map coordinate frame
       obsTransformed.push_back(transformObservation(p, observation));
     }

     vector<LandmarkObs> predicted;
     for (const auto & map_landmark : map_landmarks.landmark_list) {
       auto x_delta = ::fabs(p.x - map_landmark.x_f);
       auto y_delta = ::fabs(p.y - map_landmark.y_f);
       if (x_delta <= sensor_range && y_delta <= sensor_range) {
         predicted.push_back(LandmarkObs(map_landmark));
       }
     }

     double prob = 1.0;
     // Now we have the observations in map frame from the point of view of our particle
     // Let's figure out which landmark in our map is closest to each observation
     for (const auto & observation : obsTransformed) {
       double closest_distance = std::numeric_limits<double>::max();
       const LandmarkObs * closest = nullptr;
       for (const auto & prediction : predicted) {
         double distance = dist(observation.x, observation.y, prediction.x, prediction.y);
         if (distance < closest_distance) {
           closest_distance = distance;
           closest = &prediction;
         }
       }

       // Used for debugging
       if (closest != nullptr) {
         associations.push_back(closest->id);
         sense_x.push_back(observation.x);
         sense_y.push_back(observation.y);

         // Now that we know which landmark is closest to each observation, let's multiply
         // all of the probabilities together to get our final probability
         prob *= gaussian2D(observation.x, observation.y, closest->x, closest->y, std_landmark[0], std_landmark[1]);
       }
     }
     // If we had no observations, clear our prob estimate
     if (obsTransformed.size() == 0) {
       prob = 0.0;
       std::cout << "No observations!" << std::endl;
     }
     p.weight = prob;
     weightSum += p.weight;
     SetAssociations(p, associations, sense_x, sense_y);
   }

   if (debug) std::cout << "weightSum = " << weightSum << std::endl;
   // printStatistics();

   // Normalize the weights
   if (weightSum != 0.0) {
     for (auto & p : particles) {
       p.weight = p.weight / weightSum;
       if (debug) std::cout << p << std::endl;
     }
   } else {
     std::cout << "Weights sum to zero!" << std::endl;
   }
}

void ParticleFilter::resample() {
   double pdfmax = 0.0;
   double beta = 1.0;
   for (auto & p : particles) {
     if (p.weight > pdfmax) {
       pdfmax = p.weight;
     }
   }

   std::vector<Particle> resampled;

   std::uniform_int_distribution<unsigned> discrete_distribution(0,kNumParticles);
   std::uniform_real_distribution<double> real_distribution(0.0,1.0);
   for (size_t i = 0; i < kNumParticles; ++i) {
     auto index = discrete_distribution(generator);
     beta += real_distribution(generator) * 2.0 * pdfmax;
     while (particles[index].weight < beta) {
       beta -= particles[index].weight;
       index = (index + 1) % kNumParticles;
     }
     resampled.push_back(particles[index]);
   }
   particles = resampled;

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

double ParticleFilter::predictXAxis(double x0, double theta0, double velocity, double yaw_rate, double delta_t, double std) {
  if (yaw_rate == 0.0) {
    yaw_rate = 1.0e-10; //avoid divide by zero
  }
  double x = x0 + (velocity / yaw_rate) * (::sin(theta0 + yaw_rate * delta_t) - ::sin(theta0));
  return std::normal_distribution<double>(x, std)(generator);
}

double ParticleFilter::predictYAxis(double y0, double theta0, double velocity, double yaw_rate, double delta_t, double std) {
  if (yaw_rate == 0.0) {
    yaw_rate = 1.0e-10; //avoid divide by zero
  }
  double y = y0 + (velocity / yaw_rate) * (::cos(theta0) - ::cos(theta0 + yaw_rate * delta_t));
  return std::normal_distribution<double>(y, std)(generator);
}

double ParticleFilter::predictTheta(double theta0, double yaw_rate, double delta_t, double std) {
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
  double exponent = ::pow(x1 - x2, 2) / (2 * ::pow(sigmax, 2)) + ::pow(y1 - y2, 2) / (2 * pow(sigmay, 2));
  return (1.0 / (2 * M_PI * sigmax * sigmay)) * ::exp(-exponent);
}

void ParticleFilter::printStatistics() const {
  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::min();
  double accum = 0.0;
  static unsigned iteration = 0;
  unsigned count = 0;
  for (const auto & p : particles) {
    if (p.weight < min) {
      min = p.weight;
    }
    if (p.weight > max) {
      max = p.weight;
    }
    accum += p.weight;
    ++count;
  }
  std::cout << iteration++ << " min = " << min << " max = " << max << " average = " << accum / count << std::endl;
}
