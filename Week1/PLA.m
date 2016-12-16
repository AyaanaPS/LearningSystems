% Sets the number of random points
N = 10;
numRuns = 0;
totalUpdates = 0;
totalProbability = 0;

while (numRuns < 1000)

    % Creates the dataset of random points
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

    % Initializes the weight matrix
    weight = zeros(1,3);

    % Initially, the set of misclassified points is all the points
    misClass = points;
    % Similarily the solution matrix for misclassified points has all the
    % solutions.
    yVals = yMatrix;
    % This is a counter for the number of times we update the weight
    numUpdates = 0;

    % We continue to iterate until there are no more misclassified points
    while(isempty(misClass) ~= 1)
    
        % First we choose a random misclassified point
        temp = size(misClass);
        numMiss = temp(1);
        pointN = randi([1 numMiss], 1, 1);
        pointChosen = misClass(pointN, :);
  
        % Double check that it is indeed misclassified
        transposedWeight = transpose(weight);
        matrixResult = pointChosen * transposedWeight;
        if(sign(matrixResult) ~= yVals(pointN))
            numUpdates = numUpdates + 1;
            weight = weight + yVals(pointN)*pointChosen;
        end
    
        % Recompute the misclassified points
        newMisClass = [];
        newYVals = [];
    
        for i=1:N;
            point = points(i,:);
            transposedWeight2 = transpose(weight);
            matrixResult2 = point * transposedWeight2;
            if(sign(matrixResult2) ~= yMatrix(i))
                newMisClass = [newMisClass ; point];
                newYVals = [newYVals ; yMatrix(i)];
            end
        end
    
        misClass = newMisClass;
        yVals = newYVals;
    end

    % Add the number of updates to the total found so that we can average
    totalUpdates = totalUpdates + numUpdates;
    numRuns = numRuns + 1;
    
    % This computes the probability that f(x) and g(x) differ by
    % placing 500 test points and seeing how many times the classifications
    % with f(x) differ with the classifications for g(x).
    
    testPoints = -1 + (2)*rand(500, 3);
    numDifferences = 0;
    finalTranspose = transpose(weight);
    
    for i=1:500;
        point = testPoints(i,:);
        value = (B(1)-A(1))*(point(3)-A(2))-(B(2)-A(2))*(point(2)-A(1));
        position = sign(value);
        
        pointMatrix = point * finalTranspose;
        if(sign(pointMatrix) ~= position)
            numDifferences = numDifferences + 1;
        end
    end
    
    % The probability is added to the total found so that we can average
    runProbability = numDifferences/500;
    totalProbability = totalProbability + runProbability;
end

% This computes and displays the average updates and the average
% probability for the 1000 runs.
averageUpdates = totalUpdates/1000;
averageUpdates

finalProbability = totalProbability/1000;
finalProbability
