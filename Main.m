clear all
clc

fnames = dir('*.mat');
n_files = length(fnames);
sizesofDB = [];

%Filtering between 0.1 and 15 Hz
b = fir1(12,[0.0002,0.03]);
Hd = b'*b;

disp('Subject number:');
for i = 1:10
    
    disp(i);
    
    sub = struct2cell(load(fnames(i).name));
    
    sub_train = sub{:}.train(1:9,:)';
    sub_train_labels = sub{:}.train(11,:)';
    sub_test = sub{:}.test(1:9,:)';
    sub_test_labels = sub{:}.test(11,:)';
    
    sizesofDB = [sizesofDB; length(sub_train), length(sub_test)];
    
    %Filter data
    sub_train = filter2(Hd, sub_train);
    sub_test = filter2(Hd, sub_test);
    
    %Wavelet analysis
    [c,l] = wavedec2(sub_train,3,'haar');
    sub_train = wrcoef2('d',c,l,'haar',3); clear c l;
    [c,l] = wavedec2(sub_test,3,'haar');
    sub_test = wrcoef2('d',c,l,'haar',3); clear c l;
    
    %Decision Trees:
    tic;
    tree = fitctree(sub_train, sub_train_labels); time_tree_train(i) = toc; tic;
    labels_predict_tree = predict(tree,sub_test); time_tree_test(i) = toc;
    temp_tree = (sub_test_labels + labels_predict_tree);
    accuracy_tree(i) = (length(temp_tree(temp_tree==2)) + length(temp_tree(temp_tree==0)))/length(temp_tree)*100;
    
    %SVM:
    tic;
    svm = fitcsvm(sub_train, sub_train_labels); time_svm_train(i) = toc; tic;
    labels_predict_svm = predict(svm, sub_test); time_svm_test(i) = toc;
    temp_svm = (sub_test_labels + labels_predict_svm);
    accuracy_svm(i) = (length(temp_svm(temp_svm==2)) + length(temp_svm(temp_svm==0)))/length(temp_svm)*100;
    
end

figure(1);
plot(1:10, accuracy_tree, 'b', 1:10, accuracy_svm, 'r');
legend('DT', 'SVM'); title('Accuracy');

figure(2);
plot(1:10, time_tree_train, 'b', 1:10, time_svm_train, 'r');
legend('DT', 'SVM'); title('Train time');

figure(3);
plot(1:10, time_tree_test, 'b', 1:10, time_svm_test, 'r');
legend('DT', 'SVM'); title('Test time');