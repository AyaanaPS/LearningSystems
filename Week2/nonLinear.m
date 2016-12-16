% Number of points for training data
N = 1000;

% Number of repetitions of simulation
numRuns = 0;

% Problem 8: Ein Sum over all the repetitions to get the average.
EinSum = 0;

% Problem 9: Calculating the average number misclassified for each option
% in question 9. 
%{
weight1 = [-1; -0.05; 0.08; 0.13; 1.5; 1.5];
Ein1 = 0;
weight2 = [-1; -0.05; 0.08; 0.13; 1.5; 15];
Ein2 = 0;
weight3 = [-1; -0.05; 0.08; 0.13; 15; 1.5];
Ein3 = 0;
weight4 = [-1; -1.5; 0.08; 0.13; 0.05; 0.05];
Ein4 = 0;
weight5 = [-1; -0.05; 0.08; 1.5; 0.15; 0.15];
Ein5 = 0;
%}

while(numRuns < 1000)
    
    % Generating the N training points.
    points = -1 + (2)*rand(N, 3);
    % Creating the point matrix for question 9.
    nonLinPoints = zeros(N, 6);
    for i=1:N;
        points(i,1) = 1;
        point = points(i,:);
        % Filling the point matrix for question 9.
        nonLinPoints(i,:) = [1, point(2), point(3), point(2)*point(3), point(2)^2, point(3)^2];
    end
    
    % Computing the solution matrix using the given target function.
    yMatrix = zeros(N,1);
    for i=1:N;
        point=points(i,:);
        yMatrix(i) = sign((point(2)^2)+(point(3)^2)-0.6);
        % Flipping the value of every 10th point to simulate noise.
        if(mod(i, 10) == 0)
            if(yMatrix(i) == 1)
                yMatrix(i) = -1;
            elseif(yMatrix(i) == -1)
                yMatrix(i) = 1;
            end
        end
    end
    
    % Generating the weight matrix using linear regression
    pseudoPoints = pinv(points);
    weight = pseudoPoints * yMatrix;
    
    % Answering number 8.
    
    % Counting the number of misclassified in sample points.
    misclassified = 0;
    for i=1:N;
        point = points(i,:);
        pointResult = point * weight;
        if(sign(pointResult) ~= yMatrix(i))
            misclassified = misclassified + 1;
        end
    end
    
    % Finding the fraction of misclassified points and adding it to the
    % sum.
    Ein = misclassified/N;
    EinSum = EinSum + Ein;
    
    % Answering number 9 by computing Ein for each hypothesis option in question 9.
    %{
    misClass1 = 0;
    misClass2 = 0;
    misClass3 = 0;
    misClass4 = 0;
    misClass5 = 0;
    for i=1:N;
        nlPoint = nonLinPoints(i,:);
        pR1 = nlPoint * weight1;
        pR2 = nlPoint * weight2;
        pR3 = nlPoint * weight3;
        pR4 = nlPoint * weight4;
        pR5 = nlPoint * weight5;
        if(sign(pR1) ~= yMatrix(i))
            misClass1 = misClass1 + 1;
        end
        if(sign(pR2) ~= yMatrix(i))
            misClass2 = misClass2 + 1;
        end
        if(sign(pR3) ~= yMatrix(i))
            misClass3 = misClass3 + 1;
        end
        if(sign(pR4) ~= yMatrix(i))
            misClass4 = misClass4 + 1;
        end
        if(sign(pR5) ~= yMatrix(i))
            misClass5 = misClass5 + 1;
        end
    end
    %}
    
    % Question 9: Computing the Ein for each weight 
    %{
    E1 = misClass1/N;
    Ein1 = Ein1 + E1;
    E2 = misClass2/N;
    Ein2 = Ein2 + E2;
    E3 = misClass3/N;
    Ein3 = Ein3 + E3;
    E4 = misClass4/N;
    Ein4 = Ein4 + E4;
    E5 = misClass5/N;
    Ein5 = Ein5 + E5;
    %}

    numRuns = numRuns + 1;
end

% Computing the average Ein. 
EinFinal = EinSum/1000;

% Question 9: Checking the Ein for each weight option.
%{
E1Final = Ein1/1000;
E2Final = Ein2/1000;
E3Final = Ein3/1000;
E4Final = Ein4/1000;
E5Final = Ein5/1000;
%}
