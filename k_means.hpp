#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits.h> //for INT_MIN INT_MAX
#include <map>
#include <stdio.h>
#include <string.h>
#include <string>
#include <time.h> //for srand
#include <vector>

using namespace std;

typedef struct Node {
  int minIndex; // the index of each node
  double minDist;
  Node(int idx, double dist) : minIndex(idx), minDist(dist) {}
} tNode;

template <typename T> class KMEANS {
private:
  vector<vector<T>> dataSet; // the data set
  vector<T> mmin, mmax;
  // colLen:the dimension of vector; rowLen:the number of vectors
  int colLen, rowLen;
  int k;
  vector<vector<T>> centroids;
  vector<int> centroids_num;

  typedef struct MinMax {
    T Min;
    T Max;
    MinMax(T min, T max) : Min(min), Max(max) {}
  } tMinMax;

  vector<tNode> clusterAssment;

  /*split line into numbers*/
  void split(char *buffer, vector<T> &vec);
  tMinMax getMinMax(int idx);
  void setCentroids(tMinMax &tminmax, int idx);
  void initClusterAssment();
  double distEclud(vector<T> &v1, vector<T> &v2);

public:
  KMEANS(int k);
  KMEANS(int k, int colLen, int rowLen, vector<vector<T>> dataSet);
  void randCent();
  void print();
  void kmeans();
  vector<vector<T>> get_centroids();
  vector<int> get_centroids_num();
  vector<tNode> get_clusterAssment();
};

template <typename T> vector<vector<T>> KMEANS<T>::get_centroids() {
  return centroids;
}

template <typename T> vector<int> KMEANS<T>::get_centroids_num() {
  return centroids_num;
}

template <typename T> vector<tNode> KMEANS<T>::get_clusterAssment() {
  return clusterAssment;
}

template <typename T> void KMEANS<T>::initClusterAssment() {
  tNode node(-1, -1);
  for (int i = 0; i < rowLen; i++) {
    clusterAssment.push_back(node);
  }
}

template <typename T> void KMEANS<T>::kmeans() {
  initClusterAssment();
  bool clusterChanged = true;
  // the termination condition can also be the loops less than	some number such
  // as 1000
  while (clusterChanged) {
    clusterChanged = false;
    // step one : find the nearest centroid of each point
    cout << "find the nearest centroid of each point..." << endl;
    for (int i = 0; i < rowLen; i++) {

      // skip the white/grey point
      if (dataSet[i].at(0) <= 128 && dataSet[i].at(0) >= 126 &&
          dataSet[i].at(1) <= 128 && dataSet[i].at(1) >= 126) {
        continue;
      }

      int minIndex = -1;
      double minDist = INT_MAX;
      for (int j = 0; j < k; j++) {
        double distJI = distEclud(centroids[j], dataSet[i]);
        if (distJI < minDist) {
          minDist = distJI;
          minIndex = j;
        }
      }
      if (clusterAssment[i].minIndex != minIndex) {
        clusterChanged = true;
        clusterAssment[i].minIndex = minIndex;
        clusterAssment[i].minDist = minDist;
      }
    }

    // step two : update the centroids
    cout << "update the centroids..." << endl;
    for (int cent = 0; cent < k; cent++) {
      vector<T> vec(colLen, 0);
      int cnt = 0;
      for (int i = 0; i < rowLen; i++) {
        if (clusterAssment[i].minIndex == cent) {
          ++cnt;
          // sum of two vectors
          for (int j = 0; j < colLen; j++) {
            vec[j] += dataSet[i].at(j);
          }
        }
      }

      // mean of the vector and update the centroids[cent]
      for (int i = 0; i < colLen; i++) {
        if (cnt != 0) {
          vec[i] /= cnt;
        } else {
          std::cout << "error: cluster-" << cent << ", " << i
                    << "-th field have no points";
          vec[i] = 0;
        }
        centroids[cent].at(i) = vec[i];
      }

      centroids_num.at(cent) = cnt;
    }
    print(); // update the centroids
  }

#if 0
	typename vector<tNode> :: iterator it = clusterAssment.begin();
	while( it!=clusterAssment.end() )
	{
		cout<<(*it).minIndex<<"\t"<<(*it).minDist<<endl;
		it++;	
	}
#endif
}

template <typename T> KMEANS<T>::KMEANS(int k) { this->k = k; }

template <typename T>
KMEANS<T>::KMEANS(int k, int colLen, int rowLen, vector<vector<T>> dataSet) {
  this->k = k;
  this->colLen = colLen;
  this->rowLen = rowLen;
  this->dataSet = dataSet;
}

template <typename T> void KMEANS<T>::setCentroids(tMinMax &tminmax, int idx) {
  T rangeIdx = tminmax.Max - tminmax.Min;
  for (int i = 0; i < k; i++) {
    /* generate float data between 0 and 1 */
    centroids[i].at(idx) = tminmax.Min + rangeIdx * (rand() / (double)RAND_MAX);
  }
}

// get the min and max value of the idx column
template <typename T>
typename KMEANS<T>::tMinMax KMEANS<T>::getMinMax(int idx) {
  T min, max;
  dataSet[0].at(idx) > dataSet[1].at(idx)
      ? (max = dataSet[0].at(idx), min = dataSet[1].at(idx))
      : (max = dataSet[1].at(idx), min = dataSet[0].at(idx));

  for (int i = 2; i < rowLen; i++) {
    if (dataSet[i].at(idx) < min)
      min = dataSet[i].at(idx);
    else if (dataSet[i].at(idx) > max)
      max = dataSet[i].at(idx);
    else
      continue;
  }

  tMinMax tminmax(min, max);
  return tminmax;
}

template <typename T> void KMEANS<T>::randCent() {
  // init centroids
  vector<T> vec(colLen, 0);
  for (int i = 0; i < k; i++) {
    centroids.push_back(vec);
    centroids_num.push_back(0);
  }

  // set values by column
  srand(time(NULL));
  for (int j = 0; j < colLen; j++) {
    tMinMax tminmax = getMinMax(j);
    setCentroids(tminmax, j);
  }
}

template <typename T>
double KMEANS<T>::distEclud(vector<T> &v1, vector<T> &v2) {
  T sum = 0;
  int size = v1.size();
  for (int i = 0; i < size; i++) {
    sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
  }
  return sum;
}

template <typename T> void KMEANS<T>::split(char *buffer, vector<T> &vec) {
  char *p = strtok(buffer, " \t");
  while (p != NULL) {
    vec.push_back(atof(p));
    p = strtok(NULL, " ");
  }
}

template <typename T> void KMEANS<T>::print() {
  ofstream fout;
  fout.open("/tmp/compress/k-means-out.txt");
  if (!fout) {
    cout << "file /tmp/compress/k-means-out.txt open failed" << endl;
    exit(0);
  }

#if 0
	typename vector< vector<T> > :: iterator it = centroids.begin();
	while( it!=centroids.end() )
	{
		typename vector<T> :: iterator it2 = (*it).begin();
		while( it2 != (*it).end() )
		{
			//fout<<*it2<<"\t";
			cout<<*it2<<"\t";
			it2++;	
		}
		//fout<<endl;
		cout<<endl;
		it++;
	}
#endif

  typename vector<vector<T>>::iterator it = dataSet.begin();
  typename vector<tNode>::iterator itt = clusterAssment.begin();
  for (int i = 0; i < rowLen; i++) {
    typename vector<T>::iterator it2 = (*it).begin();
    while (it2 != (*it).end()) {
      fout << *it2 << "\t";
      it2++;
    }
    fout << (*itt).minIndex << endl;
    itt++;
    it++;
  }
}