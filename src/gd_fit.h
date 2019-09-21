#include <armadillo>

arma::mat rmse(arma::mat estimate, arma::mat expected);

arma::mat linear_model(arma::mat x, arma::mat L);

arma::mat gd_fit(const arma::mat &L,
                 const arma::mat &s,
                 const float &lr = 1.0,
                 const int &max_iter = 10000,
                 arma::mat f_model(arma::mat, arma::mat) = linear_model,
                 arma::mat f_objective(arma::mat, arma::mat) = rmse);