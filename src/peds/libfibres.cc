/** @brief C++ implementation of fibre movement algorithm by Yang Chen
 *
 * Compile with
 *
 *    g++ -fPIC -shared -O3 -o libfibres.so libfibres.cc
 *
 * Reference:
 *    Yang Chen (2022): "Towards a data fusion framework: combining multi-modal
 *    data for composites certification" Technical report, DETI-PoC2-003
 */
#include <algorithm>
#include <iostream>
#include <math.h>
#include <numeric>
#include <random>
#include <sstream>
#include <stdio.h>
#include <vector>

/**
 * @brief Periodic distance in 2d rectangular domain
 *
 * Compute periodic distance between a point and a list of n other points
 *
 * @param[in] n number of other points
 * @param[in] p coordinates of the point p
 * @param[in] q other points, coordinates of point j are q_{2j}, q_{2j+1}
 * @param[in] L  linear extents of domain in both dimensions
 * @param[out] dist output array of distances
 */
extern "C" {
void dist_periodic(const unsigned int n, const double p[2], const double *q,
                   const double L[2], double *dist) {
  for (unsigned int j = 0; j < n; ++j) {
    double d2_min = 4.0 * L[0] * L[1];
    for (int offset_x = -1; offset_x <= 1; ++offset_x) {
      for (int offset_y = -1; offset_y <= 1; ++offset_y) {
        double dx = p[0] + L[0] * offset_x - q[2 * j];
        double dy = p[1] + L[1] * offset_y - q[2 * j + 1];
        double d2 = dx * dx + dy * dy;
        d2_min = d2 < d2_min ? d2 : d2_min;
      }
    }
    dist[j] = sqrt(d2_min);
  }
}
}

/** @brief Initialise random number generator
 *
 * Initialises the random number generator with a given seed and returns the
 * state. This state can then be used in subsequent calls to the random number
 * generator to ensure that the same sequence of random numbers is generated.
 *
 * @param[in] seed seed for the random number generator
 * @param[out] rng_state_length length of the state character array
 * @param[out] rng_state character array with state of the random number
 * generator
 */
extern "C" {
void initialise_rng(unsigned int seed, unsigned int &rng_state_length,
                    char *rng_state) {
  // Set random number generator state from seed
  std::mt19937 rng;
  rng.seed(seed);
  // Copy state to string stream
  std::stringstream ss;
  ss << rng;
  std::string s = ss.str();
  // Copy state to character array
  rng_state_length = s.size();
  std::copy(s.begin(), s.end(), rng_state);
}
}

/** @brief Randomly move fibres according to algorithm
 *
 * The initial positions of the fibres are given in fibre_locations. These will
 * be overwritten with the new positions.
 *
 * @param[in] n_fibres number of fibres
 * @param[in] r_fibres fibre radii
 * @param[inout] fibre_locations fibre locations
 * @param[in] num_repeats number of repeats
 * @param[in] eps_fibres minimum distance between fibres
 * @param[in] it_max_ovlap maximum number of iterations to find new position
 * @param[inout] rng_state_length length of the random number state character
 * array
 * @param[inout] rng_state character array with state of the random number
 * generator
 */
extern "C" {
void move_fibres(const unsigned int n_fibres, const double *r_fibres,
                 double *fibre_locations, const unsigned int num_repeats,
                 const double eps_fibres, const unsigned int it_max_ovlap,
                 unsigned int &rng_state_length, char *rng_state) {
  // Set random number generator state from state object
  std::mt19937 rng;
  std::stringstream in_ss;
  in_ss.write(rng_state, rng_state_length);
  in_ss >> rng;
  // randomly shuffled labels
  std::vector<unsigned int> labels(n_fibres);
  std::iota(labels.begin(), labels.end(), 0);
  std::shuffle(labels.begin(), labels.end(), rng);
  double *dist = new double[n_fibres];
  double L[2] = {1.0, 1.0};

  std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);
  bool overlap = true;
  unsigned int k = 0;
  while (overlap || (k < num_repeats)) {
    overlap = false;
    for (int j = 0; j < n_fibres; ++j) {
      double p_j[2];
      p_j[0] = fibre_locations[2 * labels[j]];
      p_j[1] = fibre_locations[2 * labels[j] + 1];
      double r_j = r_fibres[labels[j]];
      dist_periodic(n_fibres, p_j, fibre_locations, L, dist);
      dist[labels[j]] = L[0] + L[1];
      std::sort(dist, dist + n_fibres);
      double ur_sc = (dist[0] + dist[1] + dist[2]) / 3.0;
      double dimin = 0;
      unsigned int counter = 0;
      unsigned int n_ovlap0 = n_fibres + 1;
      double p_j_leastworst[2];
      double p_j_new[2];
      while (dimin < eps_fibres) {
        double u_r = ur_sc * uniform_distribution(rng);
        double u_theta = 2 * M_PI * uniform_distribution(rng);
        p_j_new[0] = p_j[0] + u_r * cos(u_theta);
        p_j_new[1] = p_j[1] + u_r * sin(u_theta);
        while (p_j_new[0] < 0)
          p_j_new[0] += L[0];
        while (p_j_new[0] > L[0])
          p_j_new[0] -= L[0];
        while (p_j_new[1] < 0)
          p_j_new[1] += L[1];
        while (p_j_new[1] > L[1])
          p_j_new[1] -= L[1];
        dist_periodic(n_fibres, p_j_new, fibre_locations, L, dist);
        unsigned int n_ovlap1 = 0;
        dist[labels[j]] = L[0] + L[1];
        for (unsigned int j = 0; j < n_fibres; ++j) {
          dist[j] -= r_j - r_fibres[j];
          n_ovlap1 += (dist[j] < 0);
        }

        dimin = *std::min_element(dist, dist + n_fibres);
        if (n_ovlap1 < n_ovlap0) {
          p_j_leastworst[0] = p_j_new[0];
          p_j_leastworst[1] = p_j_new[1];
          n_ovlap0 = n_ovlap1;
        }
        counter++;
        if (counter > it_max_ovlap) {
          p_j_new[0] = p_j_leastworst[0];
          p_j_new[1] = p_j_leastworst[1];
          overlap = true;
          break;
        }
      }
      fibre_locations[2 * labels[j]] = p_j_new[0];
      fibre_locations[2 * labels[j] + 1] = p_j_new[1];
    }
    k++;
  }
  delete[] dist;

  // Copy random number state to state object
  std::stringstream out_ss;
  out_ss << rng;
  std::string s = out_ss.str();
  rng_state_length = s.size();
  std::copy(s.begin(), s.end(), rng_state);
}
}
