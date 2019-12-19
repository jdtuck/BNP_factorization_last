using ProgressMeter
function gibbs(A, partition_, sentAndReceived_,c_kappa,c_sigma,c_tau,c_alpha,c_beta,pred_average_vect,pred_average_vect_burn,idx_clustering)
    println(string("Starting Gibbs sample with ",n_iter," steps"))
    @showprogress for i in 1:n_iter

        # Update measure
        R_,V_,n_observed,slice_matrix,s_min = update_measure(partition_,sentAndReceived_,all_ind_mat,c_kappa,c_tau,c_sigma,c_alpha,c_beta)

        # Update partition
        if weighted
        partition_,sentAndReceived_ = update_partition(R_,V_,slice_matrix,Z_tilde,to_predict,pred_vect,directed,self_edge)
        else
        partition_,sentAndReceived_ = update_partition_unweighted(R_,V_,slice_matrix,Z_tilde,to_predict,pred_vect,directed,self_edge)
        end
        # Update hyperparameters
        for t in 1:n_steps_hyper
        c_kappa,c_sigma,c_tau,c_alpha,c_beta = update_parameters_neg2(c_kappa,
                                                                    c_sigma,
                                                                    c_tau,
                                                                    c_alpha,
                                                                    c_beta,
                                                                    prior_params,
                                                                    prop_params,
                                                                    R_,
                                                                    V_,
                                                                    sentAndReceived_,
                                                                    s_min)
        end
        PRINT_ = false
        for bob in 1:(skip-1)
            # Update measure
            R_,V_,n_observed,slice_matrix,s_min = update_measure(partition_,sentAndReceived_,all_ind_mat,c_kappa,c_tau,c_sigma,c_alpha,c_beta)

            # Update partition
            if weighted
            partition_,sentAndReceived_ = update_partition(R_,V_,slice_matrix,Z_tilde,to_predict,pred_vect,directed,self_edge)
            else
            partition_,sentAndReceived_ = update_partition_unweighted(R_,V_,slice_matrix,Z_tilde,to_predict,pred_vect,directed,self_edge)
            end

            # Update hyperparameters
            for t in 1:n_steps_hyper
                c_kappa,c_sigma,c_tau,c_alpha,c_beta = update_parameters_neg2(c_kappa,
                                                                        c_sigma,
                                                                        c_tau,
                                                                        c_alpha,
                                                                        c_beta,
                                                                        prior_params,
                                                                        prop_params,
                                                                        R_,
                                                                        V_,
                                                                        sentAndReceived_,
                                                                        s_min)
            end

        end


        # Store values
        n_active_list[i] = n_observed
        sort_idx = sortperm(R_,rev=true)
        sorted_R_ = R_[sort_idx]
        for j in 1:min(n_observed,K)
        activities_list[i,j] = sorted_R_[j]*sum(V_[sort_idx[j]])
        end

        # Store clustering
        if i > burn && (i-burn)%floor(Int,(n_iter-burn)/n_clusterings) == 0 && idx_clustering <= n_clusterings
        ind_clusterings[idx_clustering] = i
        for node_idx in 1:n
            weights = [sqrt(R_[c])*V_[c][node_idx] for c in sort_idx[1:min((2*K),length(R_))]]
            clusterings[node_idx,idx_clustering] = argmax(weights)
        end
        idx_clustering += 1
        end


        pred_average_vect = (i*pred_average_vect+pred_vect)/(i+1)

        # Compute auc error (l2 error in comments)
        error_list[i] = auc_pr(pred_vect,v_true)#norm(v_true-pred_vect)
        error_mean_list[i] = auc_pr(pred_average_vect,v_true)#norm(v_true-pred_average_vect)
        i_b = i-burn
        if i_b >= 0
        pred_average_vect_burn = (i_b*pred_average_vect_burn+pred_vect)/(i_b+1)
        error_mean_list_burn[i] = auc_pr(pred_average_vect_burn,v_true)#norm(v_true-pred_average_vect)
        end

        print_each = floor(Int,n_iter/20)
        # If monitoring sigma
        if monitoring_sigma && i%print_each == 0
        println(string("Progress: ", 100.0*i/n_iter,"%"))
        println(string("Current sigma: ",c_sigma))
        println(string("Length R: ",length(R_)))

        # Save Variables
        results_path = string("results/",data_name,"/",result_folder,'/',current_dir,"/variables/")
        mkpath(results_path)
        cd(results_path)

        # Store the variables
        save("variables.jld","activities_list",activities_list,
                            "n_active_list",n_active_list,
                            "kappa_list",kappa_list,
                            "sigma_list",sigma_list,
                            "tau_list",tau_list,
                            "alpha_list",alpha_list,
                            "beta_list",beta_list,
                            "partition_",partition_,
                            "sentAndReceived_",sentAndReceived_,
                            "clusterings",clusterings,
                            "ind_clusterings",ind_clusterings,
                            "n_iter",n_iter,
                            "skip",skip,
                            "prop_params",prop_params,
                            "prior_params",prior_params)
        cd(main_dir)

        #PRINT_ = true
        end

        kappa_list[i] = c_kappa
        sigma_list[i] = c_sigma
        tau_list[i] = c_tau
        alpha_list[i] = c_alpha
        beta_list[i] = c_beta

    end
    return kappa_list, sigma_list, tau_list, alpha_list, beta_list
end