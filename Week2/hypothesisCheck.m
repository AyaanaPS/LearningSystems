numRuns = 0;
N = 1000;

% This is the hypothesis from question 9.
weightBest = [-1; -0.05; 0.08; 0.13; 1.5; 1.5];
EoutSum =0;

while(numRuns < 1000)
    
    % Generating the 1000 out of sample points.
    points = -1 + (2)*rand(N, 3);
    % Creating the 6 element point matrix
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
    
    misclassified = 0;
    % Iterating through all 1000 points to classify it against the weight
    % from question 9
    for i=1:N;
        nonLinPoint = nonLinPoints(i, :);
        pointResult = nonLinPoint * weightBest;
        if(sign(pointResult) ~= yMatrix(i))
            misclassified = misclassified + 1;
        end
    end
    
    % Computing Eout and adding it to the sum
    Eout = misclassified/N;
    EoutSum = EoutSum + Eout;
    
    numRuns = numRuns + 1;
    
end

% Computing the final average Eout based on the Eout from all the runs.
EoutFinal = EoutSum/1000;
