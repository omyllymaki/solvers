#include "lr_finder.h"
#include "../logging/easylogging++.h"

template <class T>
LrFinder<T>::LrFinder(T solver)
{
    m_solver = solver;
};

template <class T>
double LrFinder<T>::find_best_lr(arma::mat lr_array, const arma::mat &signal)
{

    arma::mat objective_values;
    for (size_t i = 0; i < lr_array.n_elem; i++)
    {
        auto solver = m_solver;
        solver.set_learning_rate(lr_array[i]);
        arma::mat result = solver.solve(signal);
        arma::mat obj = solver.get_objective_value();
        LOG(DEBUG) << "lr " << lr_array[i] << ": objective " << obj;
        objective_values.insert_cols(i, obj);
    }

    int min_index = objective_values.index_min();
    double best_lr = lr_array[min_index];
    LOG(DEBUG) << "Best lr " << best_lr;
    
    return arma::as_scalar(best_lr);
}