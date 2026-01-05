#ifndef REGRESSION_H
#define REGRESSION_H

#include <cstdlib>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>

struct Record {
    double M;
    double N;
    double K;
    double time;
};

using namespace std;

class Regression {
    private:
        double beta = 0.0;
        double alpha = 0.0;

        void init_regression(string filename) {
            vector<Record> data;
            ifstream file(filename);

            if (!file.is_open()) {
                cerr << "Errore: Impossibile aprire il file dati.csv" << std::endl;
                exit(EXIT_FAILURE);
            }

            string line;
            getline(file, line);
            while (std::getline(file, line)) {
                std::vector<std::string> row = split(line, ',');
                if (row.size() < 6) continue;

                Record r;
                r.M = stod(row[2]); 
                r.N = stod(row[3]);
                r.K = stod(row[4]);
                r.time = stod(row[5]);

                data.push_back(r);
            }

            //Time = alpha + beta * (M*N*K)
            double sum_x = 0.0;
            double sum_y = 0.0;
            double sum_xy = 0.0;
            double sum_x2 = 0.0;
            double n_samples = static_cast<double>(data.size());

            for (const auto& r : data) {
                double x = r.M * r.N * r.K; 
                double y = r.time;

                sum_x += x;
                sum_y += y;
                sum_xy += (x * y);
                sum_x2 += (x * x);
            }

            double denominator = (n_samples * sum_x2 - sum_x * sum_x);

            if (denominator == 0) {
                cerr << "[REGRESSION] ERROR denominator\n";
                exit(EXIT_FAILURE);
            }

            double beta = (n_samples * sum_xy - sum_x * sum_y) / denominator;
            double alpha = (sum_y - beta * sum_x) / n_samples;

            cout << "\n\nREGRESSION " << filename << " ------------------------\n";
            cout << "Time = " << alpha << " + " << beta << " * (M*N*K)\n";
            cout << "Overhead base (alpha): " << alpha << " ms\n";
            cout << "Tempo per operazione (beta): " << beta << " ms/FLOP_block\n\n";
        }

        vector<string> split(const string &s, char delimiter) {
            vector<string> tokens;
            string token;
            istringstream tokenStream(s);

            while (std::getline(tokenStream, token, delimiter)) {
                tokens.push_back(token);

            }
            return tokens;
        }

    public: 
        Regression(string filename) {
            init_regression(filename);
        }

        double predict (int M, int N, int K) {
            double complexity = (double)M * N * K;
            return alpha + (beta * complexity);
        }
};
#endif
