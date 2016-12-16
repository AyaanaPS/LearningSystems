% Part 5 & 6: N = 100
% Part 7: 10

N = 100;
numRuns = 0;

% Calculate the Ein for Part 5
EinSum = 0;
% Calculate the Eout for Part 6
EoutSum = 0;
% Calculate the numUpdates for Part 7
totalUpdates = 0;

while(numRuns < 1000)
    
    % Creates the dataset of N random points
    points = -1 + (2)*rand(N, 3);
    for i=1:N;
        points(i,1) = 1;
    end

    % Creates the two points that make up the lines
    A = -1+(2)*rand(1,2);
    B = -1+(2)*rand(1,2);
    
    % Initializes the solution matrix
    yMatrix = zeros(N,1);
    
    % Fills the solution matrix by calculating the determinant
    for i=1:N;
        point = points(i,:);
        value = (B(1)-A(1))*(point(3)-A(2))-(B(2)-A(2))*(point(2)-A(1));
        pos = sign(value);
        yMatrix(i) = pos;
    end
    
    % Carrying out the linear regression algorithm:
    
    % Calculates the pseudo inverse of the points matrix
    pseudoPoints = pinv(points);
    % Calculates the weight matrix using the pseudo inverse and the solution matrix 
    weight = pseudoPoints * yMatrix;
    
    % Answering number 5
    misclassified = 0;
    % Iterating through all the points
    for i=1:N;
        point = points(i,:);
        pointResult = point * weight;
        % Checking if they are classified
        if(sign(pointResult) ~= yMatrix(i))
            % If not, add 1 to the misclassified count.
            misclassified = misclassified + 1;
        end
    end
    
    % Ein for this run is the number of misclassified points divided by 
    %   the number of points.
    Ein = misclassified/N;
    % Add this Ein to the sum to calculate the average over all the runs.
    EinSum = EinSum + Ein;
   
    % Answering number 6
    
    % Generating 1000 fresh points
    N2 = 1000;
    outSamplePoints = -1 + (2)*rand(N2, 3);
    for i=1:N2;
        outSamplePoints(i,1) = 1;
    end
    
    misclassified = 0;
    % Iterating through all 1000 points to classify it against the weight
    % generated earlier in this run.
    for i=1:N2;
        point = outSamplePoints(i,:);
        value = (B(1)-A(1))*(point(3)-A(2))-(B(2)-A(2))*(point(2)-A(1));
        pos = sign(value);
        
        pointResult = point * weight;
        if(sign(pointResult) ~= pos)
            misclassified = misclassified + 1;
        end

    end
    
    % Eout for this run is the number of misclassified points divided by
    %   the number of points (out of sample points).
    Eout = misclassified/N2;
    % Add this Eout to the sum to calculate the average over all the runs.
    EoutSum = EoutSum + Eout;
    
    %Answering number 7
    
    weightNew = transpose(weight);
    % Initial set of misclassified points.
    misClass = points;
    % Initial solution matrix for misclassified points.
    yVals = yMatrix;
    % Counter for number of weight updates until convergence.
    numUpdates = 0;
    
    % Continue to iterate until there are no more misclassified points.
    while(isempty(misClass) ~= 1)
        
        % First we choose a random misclassified point
        temp = size(misClass);
        numMiss = temp(1);
        pointN = randi([1 numMiss], 1, 1);
        pointChosen = misClass(pointN, :);
       
        % Check that it is indeed misclassified
        matrixResult = pointChosen * transpose(weightNew);
        if(sign(matrixResult) ~= yVals(pointN))
            % If so, update weight and increase the number of updates.
            numUpdates = numUpdates + 1;
            weightNew = weightNew + yVals(pointN)*pointChosen;
        end
        
        % Recompute the misclassified points with the new weight.
        newMisClass = [];
        newYVals = [];
        
        for i=1:N;
            point = points(i,:);
            matrixResult = point * transpose(weightNew);
            if(sign(matrixResult) ~= yMatrix(i))
                newMisClass = [newMisClass ; point];
                newYVals = [newYVals ; yMatrix(i)];
            end
        end
        
        misClass = newMisClass;
        yVals = newYVals;
    end
    
    % Add the number of updates to the total to get the average over the
    % runs.
    totalUpdates = totalUpdates + numUpdates; 
    
    numRuns = numRuns + 1;
end

% Gets the average Ein after all the runs for Problem 5
finalEin = EinSum/1000;

% Gets the average Eout after all the runs for Problem 6
finalEout = EoutSum/1000;

% Gets the average updates after all the runs for Problem 7
averageUpdates = totalUpdates/1000;
