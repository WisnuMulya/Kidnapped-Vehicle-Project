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
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;

  // instantiate the normal distributions of car's initial position
  std::default_random_engine gen; // generate seed number for the normal distributions
  std::normal_distribution<double> N_x(x, std[0]);
  std::normal_distribution<double> N_y(y, std[1]);
  std::normal_distribution<double> N_theta(theta, std[2]);

  // generate random particles
  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    particle.id = i;
    particle.x = N_x(gen);
    particle.y = N_y(gen);
    particle.theta = N_theta(gen);
    particle.weight = 1;

    particles.push_back(particle);
    weights.push_back(particle.weight);
  }

  is_initialized = true;
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
  double new_x, new_y, new_theta;
  std::default_random_engine gen; // generate seed number for the normal distributions

  // predict each particle movement
  for (int i = 0; i < num_particles; i++) {
    Particle particle = particles[i]; // save particle to be predicted
    
    if (yaw_rate == 0) {
      new_x = particle.x + (delta_t * velocity * cos(particle.theta));
      new_y = particle.y + (delta_t * velocity * sin(particle.theta));
      new_theta = particle.theta;
    } else {
      new_x = particle.x + (velocity / yaw_rate) * (sin(particle.theta + (yaw_rate * delta_t)) - sin(particle.theta));
      new_y = particle.y + (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + (yaw_rate * delta_t)));
      new_theta = particle.theta + (yaw_rate * delta_t);
    }

    // instantiate normal distributions of predicted states
    std::normal_distribution<double> N_x(new_x, std_pos[0]);
    std::normal_distribution<double> N_y(new_y, std_pos[1]);
    std::normal_distribution<double> N_theta(new_theta, std_pos[2]);

    // add noise to prediction
    particles[i].x = N_x(gen);
    particles[i].y = N_y(gen);
    particles[i].theta = fmod(N_theta(gen), (2 * M_PI));
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for (int i = 0; i < observations.size(); i++) {
    // find nearest neighbor to observation iteratively
    LandmarkObs nearest_neighbor = predicted[0];
    
    for (int j = 0; j < predicted.size(); j++) {
      if (dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y) <
	  dist(observations[i].x, observations[i].y, nearest_neighbor.x, nearest_neighbor.y)) {
	nearest_neighbor = predicted[j];
      }
    }

    observations[i] = nearest_neighbor;
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

  // update weights for each particle
  for (int i = 0; i < num_particles; i++) {
    Particle particle = particles[i];
    
    // transform vehicle's observation coordinate system into the map's one
    vector<LandmarkObs> transformed_observations;
    
    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs obs = observations[j];
      LandmarkObs transformed_obs;
      
      transformed_obs.x = (obs.x * cos(particle.theta)) - (obs.y * sin(particle.theta)) + particle.x;
      transformed_obs.y = (obs.x * sin(particle.theta)) + (obs.y * cos(particle.theta)) + particle.y;
      transformed_observations.push_back(transformed_obs);
    }

    // predict landmarks within sensor range
    vector<LandmarkObs> predicted_landmarks;
    
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      LandmarkObs landmark;

      // check whether landmark within sensor range
      if (dist(particle.x, particle.y,
	       map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f) <= sensor_range) {
	landmark.id = map_landmarks.landmark_list[j].id_i;
	landmark.x = map_landmarks.landmark_list[j].x_f;
	landmark.y = map_landmarks.landmark_list[j].y_f;
	
	predicted_landmarks.push_back(landmark);
      }
    }

    // associate observations to landmarks
    vector<LandmarkObs> associated_landmarks(transformed_observations);
    dataAssociation(predicted_landmarks, associated_landmarks);

    // set associations to each particle
    vector<int> associations;
    vector<double> sense_x, sense_y;
    
    for (int j = 0; j < transformed_observations.size(); j++) {
      associations.push_back(associated_landmarks[j].id);
      sense_x.push_back(transformed_observations[j].x);
      sense_y.push_back(transformed_observations[j].y);
    }
    
    SetAssociations(particles[i], associations, sense_x, sense_y);

    // update weights
    particles[i].weight = 1;
    
    for (int j = 0; j < associated_landmarks.size(); j++) {
      LandmarkObs mu = associated_landmarks[j];
      LandmarkObs obs = transformed_observations[j];
      
      // Gaussian PDF of observation, given predicted associated landmark
      double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      double exponent = exp(-((pow(obs.x - mu.x, 2) / (2 * pow(std_landmark[0], 2))) +
			     (pow(obs.y - mu.y, 2) / (2 * pow(std_landmark[1], 2)))));
      double obs_p = exponent / gauss_norm;
      
      particles[i].weight = particles[i].weight * obs_p;
    }
    
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  // instantiate the discrete distribution of weights  
  std::default_random_engine gen;
  std::discrete_distribution<> D_w(weights.begin(), weights.end());

  // resample particles
  vector<Particle> resampled_particles;
  for (int i = 0; i < num_particles; i++) {
    resampled_particles.push_back(particles[D_w(gen)]);
  }

  particles = resampled_particles;
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
