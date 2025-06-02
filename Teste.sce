// --- Configuration ---
// Network Architecture
N_in = 3;      // Based on Excel screenshot showing 3 columns (A, B, C) for input
N_hidden = 10; 
N_out = 1;     

// Training Parameters
eta = 0.1;  // Start with a moderate learning rate
epsilon = 1e-6;        
max_epochs = 50000;    
num_train_runs = 2;    

base_path = "."; 
file_X_train = fullfile(base_path, "AtributosEntradaTreinamento.csv");
file_Y_train = fullfile(base_path, "SaidaDesejadaTreinamento.csv");
file_X_test = fullfile(base_path, "AtributosEntradaTeste.csv");
file_Y_test = fullfile(base_path, "SaidaDesejadaTeste.csv");
file_results = fullfile(base_path, "resultados_treinamento.csv");

// --- Activation Function and its Derivative ---
function y = sigmoid(x)
    // Basic sigmoid, potential for NaN if x is too large/small or NaN itself
    y = 1 ./ (1 + exp(-x));
endfunction

function dy = sigmoid_derivative(y)
    dy = y .* (1 - y);
endfunction

function mse = calculate_mse(Y_pred, Y_desired)
    // If Y_pred contains NaN, mse will also be NaN, which is desired for this version
    errors = Y_desired - Y_pred;
    mse = mean(errors.^2);
endfunction

function [A1, A2] = forward_pass(X_sample, W1, B1, W2, B2)
    Z1 = W1 * X_sample + B1;
    A1 = sigmoid(Z1);
    Z2 = W2 * A1 + B2;
    A2 = sigmoid(Z2);
endfunction

disp("Loading data...");
// --- CORRECTED csvRead ASSUMING SEMICOLON FIELD SEP & COMMA DECIMAL SEP ---
csv_field_sep = ';';
csv_decimal_sep = ',';

X_train_raw = csvRead(file_X_train, csv_field_sep, csv_decimal_sep, 'double');
Y_train_raw = csvRead(file_Y_train, csv_field_sep, csv_decimal_sep, 'double');
X_test_raw  = csvRead(file_X_test,  csv_field_sep, csv_decimal_sep, 'double');
Y_test_raw  = csvRead(file_Y_test,  csv_field_sep, csv_decimal_sep, 'double');

if isempty(X_train_raw) | isempty(Y_train_raw) | isempty(X_test_raw) | isempty(Y_test_raw) then
    error("Error loading one or more data files. Check paths and file contents.");
end
disp("Data files read.");

// --- Check for NaNs immediately after reading (DEBUG) ---
if sum(isnan(X_train_raw(:))) > 0 | sum(isinf(X_train_raw(:))) > 0 then
    disp(X_train_raw(1:min(5,size(X_train_raw,1)),:)); // Display first few rows if NaN
    error("NaN/Inf detected in X_train_raw! Check CSV format and csvRead parameters.");
end
if sum(isnan(Y_train_raw(:))) > 0 | sum(isinf(Y_train_raw(:))) > 0 then
    error("NaN/Inf detected in Y_train_raw! Check CSV format and csvRead parameters.");
end
disp("Initial NaN/Inf check on raw data passed.");

X_train = X_train_raw';
Y_train_desired = Y_train_raw';
X_test = X_test_raw';
Y_test_desired = Y_test_raw';

disp("Data loaded and transposed.");

[num_features_train, num_samples_train] = size(X_train);
[num_outputs_train, dummy_cols_ytrain] = size(Y_train_desired); 
[num_features_test, num_samples_test] = size(X_test);

// Validate input dimensions AFTER potential csvRead correction
if num_features_train ~= N_in then
    error(msprintf("Training input feature count (%d) does not match N_in (%d). Check CSV or N_in.", num_features_train, N_in));
end
if num_outputs_train ~= N_out then
    error(msprintf("Training output count (%d) does not match N_out (%d). Check CSV or N_out.", num_outputs_train, N_out));
end

disp(msprintf("Training data: %d samples, %d features.", num_samples_train, num_features_train));
disp(msprintf("Test data: %d samples, %d features.", num_samples_test, num_features_test));
disp(msprintf("Learning rate (eta): %f", eta));

results_summary = []; 

for run_num = 1:num_train_runs
    mprintf("\n--- Starting Training Run %d/%d ---\n", run_num, num_train_runs);
    run_failed_due_to_nan_inf = %f; // Scilab false

    rand('seed', getdate('s') + run_num);

    W1 = rand(N_hidden, N_in) - 0.5;
    B1 = rand(N_hidden, 1) - 0.5;
    W2 = rand(N_out, N_hidden) - 0.5;
    B2 = rand(N_out, 1) - 0.5;

    epoch_errors_sse = []; 
    actual_epochs_this_run = 0; // Renamed for clarity

    for epoch = 1:max_epochs
        current_epoch_sse = 0; 
        actual_epochs_this_run = epoch; // Track current epoch number for this run

        for i = 1:num_samples_train
            X_sample = X_train(:, i);
            Y_d_sample = Y_train_desired(:, i);

            // --- Forward Propagation ---
            [A1, A2] = forward_pass(X_sample, W1, B1, W2, B2);

            // Check for NaN/Inf after forward pass (can cause SSE to be NaN)
            if sum(isnan(A2(:))) > 0 | sum(isinf(A2(:))) > 0 | sum(isnan(A1(:))) > 0 | sum(isinf(A1(:))) > 0 then
                mprintf("  WARNING: NaN/Inf in activations at Run %d, Epoch %d, Sample %d. Training might fail.\n", run_num, epoch, i);
                // We don't break here yet, let SSE calculation show NaN
            end

            error_output = Y_d_sample - A2;
            sample_sse = sum(error_output.^2);

            // Check if SSE for this sample is NaN/Inf
            if isnan(sample_sse) | isinf(sample_sse) then
                mprintf("ERROR: NaN/Inf in sample_sse at Run %d, Epoch %d, Sample %d. Stopping this run.\n", run_num, epoch, i);
                current_epoch_sse = %nan; // Mark epoch SSE as NaN
                run_failed_due_to_nan_inf = %t;
                break; // Break from sample loop
            end
            current_epoch_sse = current_epoch_sse + sample_sse;

            // --- Backward Propagation ---
            delta2 = error_output .* sigmoid_derivative(A2);
            error_hidden = W2' * delta2;
            delta1 = error_hidden .* sigmoid_derivative(A1);

            // Check for NaN/Inf in deltas (can cause weights to become NaN)
            if sum(isnan(delta1(:))) > 0 | sum(isinf(delta1(:))) > 0 | sum(isnan(delta2(:))) > 0 | sum(isinf(delta2(:))) > 0 then
                 mprintf("ERROR: NaN/Inf in deltas at Run %d, Epoch %d, Sample %d. Stopping this run.\n", run_num, epoch, i);
                 current_epoch_sse = %nan; // Mark epoch SSE as NaN, though it might already be
                 run_failed_due_to_nan_inf = %t;
                 break; // Break from sample loop
            end

            // --- Update Weights and Biases ---
            W2 = W2 + eta * delta2 * A1';
            B2 = B2 + eta * delta2;
            W1 = W1 + eta * delta1 * X_sample';
            B1 = B1 + eta * delta1;
            
            // Check weights (optional, can be verbose)
            // if sum(isnan(W1(:))) > 0 then mprintf("NaN in W1!\n"); break; end
        end // End of loop over samples
        
        if run_failed_due_to_nan_inf then
            // current_epoch_sse should be %nan if failure occurred within sample loop
            epoch_errors_sse = [epoch_errors_sse; current_epoch_sse]; // Store NaN SSE
            mprintf("Run %d, Epoch %d: Training loop failed due to NaN/Inf.\n", run_num, epoch);
            break; // Break from epoch loop
        end

        epoch_errors_sse = [epoch_errors_sse; current_epoch_sse];
        if modulo(epoch, 200) == 0 | epoch == 1 | epoch == max_epochs then 
             mprintf("Run %d, Epoch %d/%d, SSE_train: %f\n", run_num, epoch, max_epochs, current_epoch_sse);
        end

        if current_epoch_sse < epsilon then // Check if current_epoch_sse is not NaN
            mprintf("Stopping criterion met at epoch %d. SSE = %f < %f\n", epoch, current_epoch_sse, epsilon);
            break;
        end
    end // End of loop over epochs

    final_sse_this_run = epoch_errors_sse($); // Get the last SSE (could be NaN)
    mprintf("Training Run %d finished after %d epochs with Final SSE_train: %s\n", run_num, actual_epochs_this_run, string(final_sse_this_run));

    // --- Plot SSE vs Epoch (Original "NaN" plotting behavior) ---
    scf(run_num); 
    clf;
    // Plot epochs up to where they were completed or failed
    // If actual_epochs_this_run is 0 (e.g. failed on first sample of first epoch), plot nothing or a point
    if actual_epochs_this_run > 0 & ~isempty(epoch_errors_sse) then
        plot(1:length(epoch_errors_sse), epoch_errors_sse);
    else
        plot(0,0); // Placeholder if no epochs to plot
    end
    title(msprintf("Run %d: SSE vs. Epoch (Final SSE: %s)", run_num, string(final_sse_this_run)));
    xlabel("Epoch"); ylabel("Sum of Squared Error (SSE)"); xgrid;

    mse_train = %nan; mse_test = %nan;
    // Only calculate MSE if training didn't result in NaN weights/SSE
    if ~isnan(final_sse_this_run) then
        Y_pred_train = zeros(N_out, num_samples_train);
        for i = 1:num_samples_train
            [dummy_A1_train, Y_pred_train(:,i)] = forward_pass(X_train(:,i), W1, B1, W2, B2);
        end
        mse_train = calculate_mse(Y_pred_train, Y_train_desired);
        
        Y_pred_test = zeros(N_out, num_samples_test);
        for i = 1:num_samples_test
            [dummy_A1_test, Y_pred_test(:,i)] = forward_pass(X_test(:,i), W1, B1, W2, B2);
        end
        mse_test = calculate_mse(Y_pred_test, Y_test_desired);
    else
        mprintf("Run %d: MSE calculation skipped due to NaN/Inf in final SSE.\n", run_num);
    end
    mprintf("Run %d: Final MSE on Training Data: %s\n", run_num, string(mse_train));
    mprintf("Run %d: Final MSE on Test Data: %s\n", run_num, string(mse_test));
    
    results_summary = [results_summary; run_num, actual_epochs_this_run, final_sse_this_run, mse_train, mse_test];
end // End of loop over training runs

// --- Display and Save Summary Results ---
disp(" "); disp("--- Overall Training Summary ---");
disp("Run | Epochs | Final SSE (Train) | MSE (Train) | MSE (Test)");
disp("-------------------------------------------------------------");
for i_row = 1:size(results_summary, "r")
    mprintf("%3d | %6d | %17s | %11s | %10s\n", ... 
            results_summary(i_row,1), results_summary(i_row,2), ...
            string(results_summary(i_row,3)), string(results_summary(i_row,4)), string(results_summary(i_row,5)));
end

header_str = "Run,Epochs,Final_SSE_Train,MSE_Train,MSE_Test";
mprintf("\nAttempting manual CSV save to: %s\n", file_results);
fid = mopen(file_results, 'wt');
if fid == -1 then
    warning('Could not open results file for writing.');
else
    mfprintf(fid, "%s\n", header_str); 
    for i_row = 1:size(results_summary, "r")
        str_row = sprintf("%d,%d,", results_summary(i_row,1), results_summary(i_row,2));
        
        val_sse = results_summary(i_row,3);
        if isnan(val_sse) then str_row = str_row + "NaN,"; 
        elseif isinf(val_sse) then str_row = str_row + "Inf,";
        else str_row = str_row + sprintf("%.6e,", val_sse); end
        
        val_mse_train = results_summary(i_row,4);
        if isnan(val_mse_train) then str_row = str_row + "NaN,";
        elseif isinf(val_mse_train) then str_row = str_row + "Inf,";
        else str_row = str_row + sprintf("%.6f,", val_mse_train); end
        
        val_mse_test = results_summary(i_row,5);
        if isnan(val_mse_test) then str_row = str_row + "NaN";
        elseif isinf(val_mse_test) then str_row = str_row + "Inf";
        else str_row = str_row + sprintf("%.6f", val_mse_test); end
        
        mfprintf(fid, "%s\n", str_row);
    end
    mclose(fid);
    mprintf("Summary results manually saved to: %s\n", file_results);
end

disp("Script finished.");
