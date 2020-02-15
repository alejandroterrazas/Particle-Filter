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
#include <iostream>
#include <chrono>

using std::string;
using std::vector;

vector<Particle> particles;

//multivariate probability calculate the likelihood of the measurement.
//used for particle weighting
double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
    
  return weight;
}

//initialize the filter wht a set of random particles
void ParticleFilter::init(double x, double y, double theta, double std[]) {
  
  std::default_random_engine generator;
  std::normal_distribution<double> x_dist(x,std[0]);
  std::normal_distribution<double> y_dist(y,std[1]);
  std::normal_distribution<double> theta_dist(theta,std[2]); 
 
  num_particles = 100;  //numbers tried 10, 100, 500, 1000. 
  
  //generate num_particles at random locations centered on x, y with direction theta 
  //set initial weights to 1.0
  for (int i = 0; i < num_particles; i++) {
      Particle particle;  
      particle.x = x_dist(generator);
      particle.y = y_dist(generator);
      particle.theta = theta_dist(generator); 
      particle.weight = 1.0; 
      particles.push_back(particle);
      
  } 
  //set initialization flag
  is_initialized = true;
 
}


//predict the next location of every particle after dalta_t 
void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  
   //genereate
   std::default_random_engine generator;
   std::normal_distribution<double> x_noise(0,std_pos[0]);
   std::normal_distribution<double> y_noise(0,std_pos[1]);
   std::normal_distribution<double> theta_noise(0,std_pos[2]); 
   //ensure that yaw_rate is never 0 (to avoid divide by zero, nan)
   if (yaw_rate==0) {
      yaw_rate=1e-15;
   }
   
   double delta_yaw = yaw_rate * delta_t;
   double v = velocity/yaw_rate;
   //calculate the new particle locations base on linear (velocoity) and angular velocity (yaw_rate)
   for(unsigned int i=0; i < particles.size(); i++) {
      Particle p = particles.at(i); 
      p.x = p.x + v*(sin(p.theta + delta_yaw) - sin(p.theta)) + x_noise(generator);
      p.y = p.y + v*(cos(p.theta) - cos(p.theta + delta_yaw)) + y_noise(generator);       
      p.theta = p.theta + delta_yaw + theta_noise(generator);    
      particles.at(i) = p;
   }
  
}

//update the weights for each particle by calculating the likelihood of observations
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
    
   double sig_x = std_landmark[0];
   double sig_y = std_landmark[1]; 
   
   //loop over particles adding the observations to each
   for (unsigned int i=0; i < particles.size(); i++) {
     
       Particle particle = particles.at(i);
       double xp = particle.x;
       double yp = particle.y;
       double theta = particle.theta;
       vector<int> associations; //vector of associations for each particle
       vector<double> weights;  //all weights for all observations
       
       //loop over the observations adding each to the currently chosen particle 
       for (unsigned int j=0; j < observations.size(); j++) {
         
          double xobs = observations.at(j).x;
          double yobs = observations.at(j).y;
         
          //eliminate observations that are outside the sensor range         
          if (sqrt(pow(xobs,2) + pow(yobs,2)) > sensor_range) {
               //std::cout << "obervation out of range" << std::endl;
               continue;
          }
              
         //map observations onto particles
          double xmap = xp + (cos(theta) * xobs) - (sin(theta) * yobs);
          double ymap = yp + (sin(theta) * xobs) + (cos(theta) * yobs); 
 
         
         //
          vector<int> distance_vals;
          vector<int> indices; 
          vector<double> distances;
        
         //calcualte difference between landmarks and particle obs
          for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); k++) {
              distance_vals.push_back(pow((map_landmarks.landmark_list.at(k).x_f - xmap),2) +
                               pow((map_landmarks.landmark_list.at(k).y_f - ymap),2));
              indices.push_back(map_landmarks.landmark_list.at(k).id_i);    
           }
 
           //find minimum distance to all landmarks 
           double findValue = *min_element(distance_vals.begin(), distance_vals.end());
           vector<int> association; 
         
           //the following puts the indices of all elements matching the minimum 
           //into associations.  Useful if you have multiple points that are tied
           std::vector<int>::iterator iter = distance_vals.begin(); 
           while ((iter = std::find(iter, distance_vals.end(), findValue)) != distance_vals.end()) {       
              int idx = std::distance(distance_vals.begin(), iter); iter++; 
              association.push_back(indices[idx]);
              distances.push_back(distance_vals[idx]);
           }
            
           //if (association.size() > 1) {
            // std::cout << "TIES ENCOUNTERED" << std::endl;
           //}
           //take the first association (if there are ties)
           int indx = association.at(0)-1;  //indices on map start at 1; 
           double mu_x  = map_landmarks.landmark_list.at(indx).x_f;
           double mu_y = map_landmarks.landmark_list.at(indx).y_f;
  
           double weight = multiv_prob(sig_x, sig_y, xmap, ymap, mu_x, mu_y);

           weights.push_back(weight);
           
     } //end loop over observations

 
     particle.associations = associations;
     
     //important, you need 1 for argument 3 or it will return 0
     double particle_weight = std::accumulate(begin(weights), end(weights), 1., 
                                     std::multiplies<double>());
        
     
   //update the weights by summing the multivariate gaussians values
   particle.weight = particle_weight;  
   particles.at(i) = particle;
   } //done looping over particle

} 

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
 
  //set up random number generator (choose a unique seed for every iteration)
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  static std::default_random_engine generator (seed);
  
  vector<double> particle_weights; 
  for (unsigned int i = 0; i < particles.size(); i++) {
      particle_weights.push_back(particles.at(i).weight);
   }
 
   double beta_max = 2*(*max_element(particle_weights.begin(), 
                                    particle_weights.end()));
  
   //choose a random particle using a uniform distribution    
   std::uniform_int_distribution<int>int_distribution(0,num_particles-1);
   int index = int_distribution(generator);
    
   //set up real number sampling for beta
   //for importance sampling select beta_max as 2*max_weight of all particles
   std::uniform_real_distribution<double>real_distribution(0,beta_max);
   double beta = 0;  //initialize beta to 0
  
   //keepsers contains the index of particles to resample
   vector<int> keepers;
   for (unsigned int i = 0; i<particles.size(); i++) {
 
     beta += real_distribution(generator);
     //loop over indices until beta is less than the width of current weight
     while (particle_weights.at(index) < beta) {
        // particle_list.push_back(index);
         beta = beta - particle_weights.at(index);
         //std::cout << "beta: " << beta << " index: " << index << std::endl;
         //std::cout << "weight: " << particle_weights.at(index);
         index += 1;
         if (index==num_particles) {
            index=0;
         }  
     } 
 
   keepers.push_back(index);  
 }  
  
 //take particle indices in keepers and make new particle list
 vector<Particle> resampled_particles;
 for (unsigned int i = 0; i < keepers.size(); i++) {
      Particle particle = particles.at(keepers.at(i));                
      resampled_particles.push_back(particle);
 }  
 //replace particles with resampled particles. 
 particles = resampled_particles;   
     
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
   //Note: I didn't use this method.
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

