import random
import pylab


verbose = False                     # whether to print out a description
                                    # of events as the program runs
                                    
numServers = 4                      # number of serving points
maxServerQueue = 10                 # max length of a server queue before the
                                    # customer gives up and leaves
numTables = 25                      # number of tables
maxTableQueue = 10                  # max length of the queue for a table
                                    # before the customer gives up and leaves


meanInterArrivalTime = 30           # in seconds, exponentially distributed 
meanServiceTime = 100               # in seconds, 2-Erlang distribution
meanEatingTime = 600                # in seconds, 2-Erlang distribution
finishTime = 36000                  # simulate 10 hours of operation


class Event:
    def __init__(self,what,when,where=None):
        self.type = what
        self.time = when
        self.where = where

class Person:
    def __init__(self,arrivalTime):
        self.arrivalTime = arrivalTime
        self.serviceTime = None
        self.totalWaitingTime = None

class Server:
    def __init__(self):
        self.status = 'free'
        self.queue = []

class Table:
    def __init__(self):
        self.status = 'free'
        self.diner = None

class TableQueue:
    def __init__(self):
        self.queue = []


# some global variables
eventList = []
servers = []
tables = []
simTime = 0.0
timeOfLastEvent = 0.0
ts = []
tq = []
qs = []
for i in range(0,numServers):
    q = []
    qs.append(q)


# furnish the restaurant with servers, tables, and a table queue
for i in range(0,numServers):
    s = Server()
    servers.append(s)

for i in range(0,numTables):
    t = Table()
    tables.append(t)

tableQueue = TableQueue()





# note that everything in the restaurant starts out empty
areaServerStatus = [0.0] * numServers
areaServerQueue = [0.0] * numServers
areaTableStatus = [0.0] * numTables
areaTableQueue = 0.0
totalArrivals = 0
totalTurnedAway = 0
totalServed = 0
totalUnseated = 0
totalSeated = 0
totalSatisfied = 0
totalDelay = 0.0

def erlang(mean,order):
    # calculate an erlang-distributed random variate
    total = 0.0
    part = mean / order
    for i in range(0,order):
        total += random.expovariate(1.0/part)
    return total

def schedule(event):
    # insert something into the event list at the right point
    for i in range(0,len(eventList)):
        if event.time < eventList[i].time:
            eventList.insert(i,event)
            break
    
def initialize():
    # Post arrival of first customer to event list
    eventList.append(Event('arrival',
                           random.expovariate(1.0 / meanInterArrivalTime )))
    # Post simulation completion to event list
    eventList.append(Event('endOfSimulation',finishTime))
    # Post far-future placeholder to event list
    eventList.append(Event('endOfSimulation',finishTime * 100))

    if verbose:
        print "First customer set to arrive at", eventList[0].time
        print "Simulation set to end at", eventList[1].time

def timing():
    # Event list is already ordered, so we take the event on the left
    global eventList
    nextEvent = eventList.pop(0)
    # Advance the simulation clock; tell Python it's a global variable
    global simTime
    simTime = nextEvent.time
    return nextEvent

def updateTimeAvgStats():
    global areaServerStatus, areaServerQueue, areaTableStatus, areaTableQueue, timeOfLastEvent
    # how long since the last thing happened?
    timeSinceLastEvent = simTime - timeOfLastEvent
    timeOfLastEvent = simTime
    # update all the stats recording area-under-the-curve
    for i in range(0,numServers):
        if ( servers[i].status == 'busy' ):
            areaServerStatus[i] += 1.0 * timeSinceLastEvent
        areaServerQueue[i] += len(servers[i].queue) * timeSinceLastEvent
    for i in range(0,numTables):
        if ( tables[i].status == 'busy' ):
            areaTableStatus[i] += 1.0 * timeSinceLastEvent
    areaTableQueue += len(tableQueue.queue) * timeSinceLastEvent
    # update some record-keeping stats on queue lengths
    global ts, qs, tq
    ts.append(simTime)
    for i in range(0,numServers):
        qs[i].append(len(servers[i].queue))
    tq.append(len(tableQueue.queue))
    
def arrival():
    global totalArrivals, servers, simTime
    if verbose:
        print simTime, ": someone arrives."
    # Increment the total number of arrival events
    totalArrivals += 1
    # Schedule the next arrival event
    nextArrival = Event('arrival',
                        simTime + random.expovariate(1.0 / meanInterArrivalTime ))
    schedule(nextArrival)
    # Note that no queue could be longer than this
    lengthOfShortestQueue = maxServerQueue + 1
    # Find the shortest server queue; new person will join it
    for i in range(0,numServers):
        if len(servers[i].queue) < lengthOfShortestQueue:
            lengthOfShortestQueue = len(servers[i].queue)
            shortQueue = i

    # Three possibilities: all queues too long and they go away, they join
    # the short queue, or the short queue is empty and they go straight to
    # the counter
    if lengthOfShortestQueue >= maxServerQueue:
        global totalTurnedAway
        totalTurnedAway += 1
        if verbose:
            print "All queues are too long and they leave immediately."
    elif lengthOfShortestQueue > 0:
        newGuy = Person(simTime)
        servers[shortQueue].queue.append(newGuy)
        if verbose:
            print "They go to place", lengthOfShortestQueue+1,
            print "in queue", shortQueue
    else:
        newGuy = Person(simTime)
        servers[shortQueue].queue.append(newGuy)
        servers[shortQueue].status = 'busy'
        # Need to schedule the completion of service event
        completion = Event('serverCompletion',
                           simTime + erlang(meanServiceTime,2),
                           shortQueue )
        schedule(completion)
        if verbose:
            print "They walk up to server", shortQueue, "and order."

def serverCompletion(server):

    global simTime, totalServed, totalUnseated, totalSeated, tableQueue, servers, tables, totalDelay

    if verbose:
        print simTime, ": person at server ", server, "collects their order."

    newlyServedGuy = servers[server].queue.pop(0)
    totalServed += 1

    # Three possibilities for the person who has just collected their order
    # 1. The queue for tables is too long and they leave
    # 2. They join the queue for tables.
    # 3. They go straight to a table.
    
    if len(tableQueue.queue) > maxTableQueue:
        totalUnseated += 1
        if verbose:
            print "They baulk at the length of the queue for tables and leave."
    elif len(tableQueue.queue) > 0:
        tableQueue.queue.append(newlyServedGuy)
        if verbose:
            print "They join the queue for tables in position", len(tableQueue.queue)
    else:
        foundATable = False
        for i in range(0,numTables):
            if tables[i].status == 'free':
                foundATable = True
                tables[i].status = 'busy'
                totalSeated += 1
                tables[i].diner = newlyServedGuy 
                finishEating = Event('tableDepart',
                                     simTime + erlang(meanEatingTime, 2),
                                     i )
                schedule(finishEating)
                newlyServedGuy.serviceTime = simTime
                totalDelay += simTime - newlyServedGuy.arrivalTime
                if verbose:
                    print "They go straight to table", i, "and sit down to eat."
                break

        if not foundATable:
            tableQueue.queue.append(newlyServedGuy)
            if verbose:
                print "All tables are full and they start a queue."

    # Meanwhile back at the servers...
    if len(servers[server].queue) == 0:
        servers[server].status = 'free'
        if verbose:
            print "Server", server, "is now idle."
    else:
        # Need to schedule the completion of service event
        completion = Event('serverCompletion',
                           simTime + erlang(meanServiceTime,2),
                           server )
        schedule(completion)
        
        if verbose:
            print "The queue for server", server, "now advances by one",
            print "and an order is placed."



def tableDepart(table):
    global totalSatisfied, totalSeated, tables, totalDelay

    if verbose:
        print simTime, ": The person at table", table, "finishes eating and leaves."

    totalSatisfied += 1
    tables[table].status = 'free'
    departingGuy = tables[table].diner
    tables[table].diner = None

    # Two possibilities: either there's someone waiting in the table queue or there isn't
    if len(tableQueue.queue) > 0:
        impatientGuy = tableQueue.queue.pop(0)
        tables[table].diner = impatientGuy
        tables[table].status = 'busy'
        totalSeated += 1
        finishEating = Event('tableDepart',
                             simTime + erlang(meanEatingTime, 2),
                             table )
        schedule(finishEating)
        impatientGuy.serviceTime = simTime
        totalDelay += simTime - impatientGuy.arrivalTime
        
        if verbose:
            print "Someone from the table queue replaces them."

def report():
    print finishTime, "units of simulated time,",
    print numServers, "servers, and", numTables, "tables."
    print "There were", totalArrivals, "arrivals;",
    print totalServed, '(%0.2f%%)' % (100.0*totalServed/totalArrivals),
    print "were served"
    print "and", totalSeated, '(%0.2f%%)' % (100.0*totalSeated/totalArrivals),
    print "found tables."
    print totalTurnedAway, "customers baulked because of long server queues."
    print totalUnseated, "could not find a table."

    print "The average waiting time before eating was",
    print '%0.2f sec.' % (totalDelay / totalSeated )

    avgServerStatus = 0.0
    avgServerQueue = 0.0
    avgTableStatus = 0.0

    for i in range(0,numServers):
        avgServerStatus += areaServerStatus[i]
        avgServerQueue += areaServerQueue[i]

    for i in range(0,numTables):
        avgTableStatus += areaTableStatus[i]

    print "Average server utilization was",
    print '%0.2f%%.' % (100.0 * avgServerStatus / ( numServers * finishTime ))
    print "Average server queue length was",
    print '%0.2f.' % (avgServerQueue / ( numServers * finishTime ))
    print "Average table utilization was",
    print '%0.2f%%.' % (100.0 * avgTableStatus / ( numTables * finishTime ))
    print "Average table queue length was",
    print '%0.2f.' % (areaTableQueue / finishTime )

    pylab.plot(ts,qs[0],'r-')
    pylab.ylabel('Queue length, server 0')
    pylab.xlabel('Time')
    pylab.show()

    pylab.plot(ts,tq,'r-')
    pylab.ylabel('Table queue length')
    pylab.xlabel('Time')
    pylab.show()

## The main section of the program

initialize()

while True:
    nextEvent = timing()
    updateTimeAvgStats()
    if nextEvent.type == 'arrival':
        arrival()
    elif nextEvent.type == 'serverCompletion':
        serverCompletion(nextEvent.where)
    elif nextEvent.type == 'tableDepart':
        tableDepart(nextEvent.where)
    elif nextEvent.type == 'endOfSimulation':
        break

report()

