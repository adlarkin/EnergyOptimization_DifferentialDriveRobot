import numpy as np
from scipy.optimize import minimize
from random import randint
from random import uniform

'''
UNITS
newton = kg*(m / s^2)
represent mass (m) in kg
distance = meters
angle = radians

m = robotBodyMass = 6.3 kg
b = robotBodyRadius = .177
r = wheel radius = .038 m
max velocity = .65 m/s
max omega (angular velocity) = pi/2 rad/s
https://www.clearpathrobotics.com/turtlebot-2-open-source-robot/
'''

global m
global b
global r
global vMax
global tauMax
global deltaTauMax
global distTolerance
global optimizationMethod
global maxIterations
# save specifications for the turtlebot
m = 6.3
b = .177
r = .038
vMax = .65

def objective(inputs, initialState, finalState, deltaT):
    """
    :type inputs:       numpy.ndarray
    :type initialState: numpy.ndarray
    :type finalState:   numpy.ndarray
    :type numSteps:     int
    :type deltaT:       float
    initialState =  [x,y,theta,v,omega,leftTorque,rightTorque]
    finalState   =  [x,y]
    """
    cost = 0
    currState = np.array(initialState)
    for torqueIndex in range(0,len(inputs),2):
        t_left = inputs[torqueIndex]
        t_right = inputs[torqueIndex+1]
        currState = updateState(currState, t_left, t_right, deltaT)
        leftWheelCost = getLeftWheelCost(currState, t_left)
        rightWheelCost = getRightWheelCost(currState, t_right)
        cost += leftWheelCost + rightWheelCost
    return cost

def getCalculatedEndLocation(torqueInputs, initialState, deltaT):
    currState = np.array(initialState)
    for i in range(0,len(torqueInputs),2):
        t_left = torqueInputs[i]
        t_right = torqueInputs[i+1]
        currState = updateState(currState, t_left, t_right, deltaT)
    xCalc = currState[0]
    yCalc = currState[1]
    return xCalc, yCalc

def getMaxTau(deltaT):
    return (r * m * vMax) / (2 * deltaT)

def getDistanceCost(goal, loc):
    x = loc[0]
    y = loc[1]
    return np.linalg.norm(goal - np.array([x,y]))

def getLeftWheelCost(robotState, leftTorque):
    """
    :type robotState: numpy.ndarray
    """
    v = robotState[3]
    omega = robotState[4]
    return abs(leftTorque * ((v + (b*omega)) / r))

def getRightWheelCost(robotState, rightTorque):
    """
    :type robotState: numpy.ndarray
    """
    v = robotState[3]
    omega = robotState[4]
    return abs(rightTorque * ((v - (b*omega)) / r))

def updateState(oldState, leftTorque, rightTorque, deltaT):
    """
    :type oldState: numpy.ndarray
    """
    x = updateX(oldState[0], oldState[3], oldState[2], deltaT)
    y = updateY(oldState[1], oldState[3], oldState[2], deltaT)
    theta = updateOrientation(oldState[2], oldState[4], deltaT)
    v = updateVelocity(oldState[3], leftTorque, rightTorque, deltaT)
    omega = updateOmega(oldState[4], leftTorque, rightTorque, deltaT)
    return np.array([x, y, theta, v, omega])

def updateX(oldX, oldVelocity, orientation, deltaT):
    return oldX + (oldVelocity*np.cos(orientation)*deltaT)

def updateY(oldY, oldVelocity, orientation, deltaT):
    return oldY + (oldVelocity*np.sin(orientation)*deltaT)

def updateOrientation(oldOrientation, omega, deltaT):
    return oldOrientation + (omega*deltaT)

def updateVelocity(oldVelocity, leftTorque, rightTorque, 
    deltaT, leftFriction=0, rightFriction=0):
    # using forward difference for now
    return oldVelocity + ((deltaT*(leftTorque+rightTorque-leftFriction-rightFriction))/(r*m))

def updateOmega(oldOmega, leftTorque, rightTorque, 
    deltaT, leftFriction=0, rightFriction=0):
    # using forward difference for now
    return oldOmega + ((deltaT/(2*m*b*r))*(rightTorque-leftTorque-rightFriction+leftFriction))

def thresholdTorque(torque):
    if torque > tauMax:
        return tauMax
    if torque < (-1 * tauMax):
        return -1 * tauMax
    return torque

def incrementTorque(prevTorque, sign, deltaT):
    changeInTau = uniform(0, deltaTauMax)
    # changeInTau = deltaTauMax
    # sign = 1
    nextTorque = prevTorque + (sign * deltaT * changeInTau)
    return thresholdTorque(nextTorque)

def randomGuessFineTuner(inputs, initialState, finalState, deltaT):
    # the cost for the initial guess is the distance
    # (we want the initial guess to get us as close to the goal location as possible)
    cost = 0
    currState = np.array(initialState)
    for torqueIndex in range(0,len(inputs),2):
        t_left = inputs[torqueIndex]
        t_right = inputs[torqueIndex+1]
        currState = updateState(currState, t_left, t_right, deltaT)
        cost += getDistanceCost(finalState, (currState[0], currState[1]))
    cost += getDistanceCost(finalState, (currState[0], currState[1]))
    return cost

def setTorqueSmoothnessConstraint(allConstraints, initialState, torques, deltaT):
    allConstraints.append({'type':'ineq','fun':torqueIncreaseConstraint,'args':(0,deltaT,initialState[5])})
    allConstraints.append({'type':'ineq','fun':torqueDecreaseConstraint,'args':(0,deltaT,initialState[5])})
    allConstraints.append({'type':'ineq','fun':torqueIncreaseConstraint,'args':(1,deltaT,initialState[6])})
    allConstraints.append({'type':'ineq','fun':torqueDecreaseConstraint,'args':(1,deltaT,initialState[6])})
    for indx in range(2,len(torques),2):
        allConstraints.append({'type':'ineq','fun':torqueIncreaseConstraint,'args':(indx,  deltaT,torques[indx-2])})
        allConstraints.append({'type':'ineq','fun':torqueDecreaseConstraint,'args':(indx,  deltaT,torques[indx-2])})
        allConstraints.append({'type':'ineq','fun':torqueIncreaseConstraint,'args':(indx+1,deltaT,torques[indx-1])})
        allConstraints.append({'type':'ineq','fun':torqueDecreaseConstraint,'args':(indx+1,deltaT,torques[indx-1])})

def setMaxAndMinTorqueBounds(n):
    torqueRange = (-tauMax, tauMax)
    bnds = (torqueRange,) * (2*n)
    return bnds

def randomizeInitialGuess(initialState, goalLocation, n, deltaT, numRandomTries):
    closestDist = np.inf
    bestGuess = [0] * (2*n)
    bestLocation = None
    for randConfig in range(numRandomTries):
        # torque can either increase, decrease, or stay the same between iterations
        leftSign = randint(-1,1)
        rightSign = randint(-1,1)
        torques = [0] * (2*n)
        # create a random initial guess
        torques[0] = incrementTorque(initialState[5], leftSign, deltaT)
        torques[1] = incrementTorque(initialState[6], rightSign, deltaT)
        for i in range(2,len(torques),2):
            leftSign = randint(-1,1)
            rightSign = randint(-1,1)
            torques[i] = incrementTorque(torques[i-2], leftSign, deltaT)
            torques[i+1] = incrementTorque(torques[i-1], rightSign, deltaT)
        # use this random initial guess to figure out how close it gets the robot to the goal
        loc = getCalculatedEndLocation(torques, initialState, deltaT)
        # update the best guess to this random initial guess if it's the best so far
        dist = getDistanceCost(goalLocation, (loc[0], loc[1]))
        if dist < closestDist:
            closestDist = dist
            bestGuess = torques
            bestLocation = loc
    # run an optimization on the initial guess to get it closer to the goal
    bnds = setMaxAndMinTorqueBounds(n)
    otherParams = (initialState, goalLocation, deltaT)
    allConstraints = []
    setTorqueSmoothnessConstraint(allConstraints, initialState, bestGuess, deltaT)
    result = minimize(randomGuessFineTuner, bestGuess, args=otherParams, method=optimizationMethod, 
        constraints=allConstraints, bounds=bnds, options=maxIterations)
    if not result.success:
        raise Exception("Optimization for the initial guess failed")
    bestGuess = result.x
    # show the final initial guess that was calculated, along with how close it got the robot to the goal
    print("initial state:")
    print(initialState)
    print("\ninitially guessed torques:")
    for i in range(0,len(bestGuess),2):
        print(bestGuess[i],"\t",bestGuess[i+1])
    print("\ncalculated location:")
    print(bestLocation[0], ",", bestLocation[1])
    print("\ndistance:")
    print(closestDist)
    return bestGuess

def endLocationConstraint(torques, initState, goal, deltaT):
    endLoc = getCalculatedEndLocation(torques, initState, deltaT)
    diff = getDistanceCost(goal, endLoc)
    return distTolerance - diff

def torqueIncreaseConstraint(torques, currTorqueIdx, deltaT, prevTorque):
    return prevTorque + (deltaT*deltaTauMax) - torques[currTorqueIdx]

def torqueDecreaseConstraint(torques, currTorqueIdx, deltaT, prevTorque):
    return torques[currTorqueIdx] - (prevTorque - (deltaT*deltaTauMax))


########################################################################################
# THESE FUNCTIONS CAN BE USED TO PLOT THE ROBOT'S BEHAVIOR BASED ON THE OPTIMIZED INPUTS
import matplotlib.pyplot as plt

def getStateSpaceValues(inputs, initialState, finalState, deltaT):
    allStates = np.zeros((int(len(inputs)/2)+1 , 5))
    allStates[0,:] = np.array(initialState)[0:5]
    currStateIndx = 1
    currState = np.array(initialState)
    for torqueIndex in range(0,len(inputs),2):
        t_left = inputs[torqueIndex]
        t_right = inputs[torqueIndex+1]
        currState = updateState(currState, t_left, t_right, deltaT)
        allStates[currStateIndx,:] = np.array(currState)
        currStateIndx += 1
    return allStates
########################################################################################





#########################################################################################
# DEFINE VARIABLES HERE
optimizationMethod = 'SLSQP'
maxIterations = {'maxiter' : 1000000}
epsilon = .01   # how much the tree can increment at a time (RRT parameter)
distTolerance = epsilon / 1000000
distTolerance = epsilon * (10 ** -7)
deltaTauMax = 1.5    # how much the torque can change per second
initState = [0, 0, (np.pi)/2, 0, 0, 0, 0]  # [x,y,theta,v,omega,leftTorque,rightTorque]
endLocation = [initState[0], initState[1] + epsilon]
# endLocation = [initState[0] + (.1*epsilon*np.cos(np.pi/4)), initState[1] + (epsilon*np.sin(np.pi/4))]
n = 30
randomTries = 30
timeStep = .01
guess = [.25] * (2*n)   # 2*n because there are 2 inputs for every n (left/right torque)   
#########################################################################################

# set the max tau and get the calculated end location based on the initial guess
tauMax = getMaxTau(timeStep)
guess = randomizeInitialGuess(initState, endLocation, n, timeStep, randomTries)

# setup bounds and constraints for the optimization
bnds = setMaxAndMinTorqueBounds(n)
otherParams = (initState, endLocation, timeStep)
allConstraints = []
allConstraints.append({'type': 'ineq', 'fun': endLocationConstraint, 'args': otherParams}) 
setTorqueSmoothnessConstraint(allConstraints, initState, guess, timeStep)

print("\nrunning the optimization...\n")
result = minimize(objective, guess, args=otherParams, method=optimizationMethod, 
    constraints=allConstraints, bounds=bnds, options=maxIterations)
print(result)
finalVals = result.x
print("\nfinal torques:")
for i in range(0,len(finalVals),2):
    print(finalVals[i],"\t\t",finalVals[i+1])
optimizedLoc = getCalculatedEndLocation(finalVals, initState, timeStep)
print("\nfinal location:",optimizedLoc)
print("location difference:",getDistanceCost(endLocation,optimizedLoc))

# print robot behavior to ensure the optimization gave reasonable results
timeLabel = "Time (s)"
allStates = getStateSpaceValues(finalVals, initState, endLocation, timeStep)
all_x_locs = allStates[:,0]
all_y_locs = allStates[:,1]
all_orientations = allStates[:,2]
all_velocities = allStates[:,3]
all_angular_velocities = allStates[:,4]
time_range = np.arange(0,((n+1)*timeStep),timeStep)
plt.subplot(231)
plt.plot(all_x_locs[1:-1], all_y_locs[1:-1], 'y.', endLocation[0], endLocation[1], 'cX', all_x_locs[0], all_y_locs[0], 'b*', all_x_locs[-1], all_y_locs[-1], 'r*')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
plt.legend(["Intermediate Positions","Target End Position","Starting Position","Calculated End Position"])
plt.title("Position Over Time")
plt.subplot(232)
plt.plot(time_range, np.append(initState[-2], finalVals[::2]), 'r', time_range, np.append(initState[-1], finalVals[1::2]), 'b')
plt.xlabel(timeLabel)
plt.ylabel("Torques (newton-meter)")
plt.legend(["Left Wheel Torques","Right Wheel Torques"])
plt.title("Torques Over Time")
plt.subplot(233)
plt.plot(time_range, all_velocities)
plt.xlabel(timeLabel)
plt.ylabel("Velocity (m/s)")
plt.title("Velocity Over Time")
plt.subplot(234)
plt.plot(time_range, all_orientations)
plt.xlabel(timeLabel)
plt.ylabel("Orientation (radians)")
plt.title("Orientation Over Time")
plt.subplot(235)
plt.plot(time_range, np.zeros(time_range.shape), 'r--', time_range, all_angular_velocities, 'b')
plt.xlabel(timeLabel)
plt.ylabel("Angular Velocity (rad/s)")
plt.title("Angular Velocity Over Time")

plt.show()