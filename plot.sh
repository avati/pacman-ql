#!/bin/bash

function plot {
    type=$1
    title="$2"
    file1="$3"
    file2="$4"
    legend1="$5"
    legend2="$6"


    file=$type.log
    bfile=$type-rote.log

    grep Progress $file1 | cut -f2 -d: | tr ',' '\n' | sed -e 's/ *//g' > values1.dat
    grep Progress $file2 | cut -f2 -d: | tr ',' '\n' | sed -e 's/ *//g' > values2.dat

    lines=$(wc -l values1.dat | cut -f1 -d' ')
    >x.dat
    >oracle.dat
    for i in $(seq $lines); do
	echo $i >> x.dat
	echo 1.0 >> oracle.dat
    done

    octave -q <<EOF
pkg load financial
x=load("x.dat");
[v,vs] = movavg(load("values1.dat"),1,50);
[b,bs] = movavg(load("values2.dat"),1,50);
o=ones($lines,1);
hold on;
plot(x,vs,"color","blue");
plot(x,bs,"color","red");
plot(x,o,"color","green");
ylim([0 1.09]);
xlabel("Number of games");
ylabel("Average score");
legend("$legend1","$legend2","Oracle");
title("$title");
print -dpng $type.png;
EOF
}


function goal1 {
    rm -f database.db && python pacman.py -p QLearningAgent -a featureExts=all -n 1000 -q >rand-large.log 2>&1
    rm -f database.db && python pacman.py -p QLearningAgent -a featureExts=rote -n 1000 -q >rand-large-rote.log 2>&1
    plot rand-large "Large Maze, Default Ghost" rand-large.log rand-large-rote.log "Q-Learning" "Baseline"


    rm -f database.db && python pacman.py -p QLearningAgent -a featureExts=all -g DirectionalGhost  -n 1000 -q >dir-large.log 2>&1
    rm -f database.db && python pacman.py -p QLearningAgent -a featureExts=rote -g DirectionalGhost  -n 1000 -q >dir-large-rote.log 2>&1
    plot dir-large "Large Maze, Directional Ghost" dir-large.log dir-large-rote.log "Q-Learning" "Baseline"


    rm -f database.db && python pacman.py -p QLearningAgent -a featureExts=all -l smallClassic -n 1000 -q >rand-small.log 2>&1
    rm -f database.db && python pacman.py -p QLearningAgent -a featureExts=rote -l smallClassic -n 1000 -q >rand-small-rote.log 2>&1
    plot rand-small "Small Maze, Default Ghost" rand-small.log rand-small-rote.log "Q-Learning" "Baseline"

    rm -f database.db && python pacman.py -p QLearningAgent -a featureExts=all -g DirectionalGhost -l smallClassic -n 1000 -q >rand-large.log 2>&1
    rm -f database.db && python pacman.py -p QLearningAgent -a featureExts=rote -g DirectionalGhost -l smallClassic -n 1000 -q >rand-large-rote.log 2>&1
    plot dir-small "Small Maze, Directional Ghost" dir-small.log dir-small-rote.log "Q-Learning" "Baseline"
}


function goal2 {
    rm -f preTrainRand.db
    python pacman.py -p QLearningAgent -a featureExts=all,db=preTrainRand -n 1000 -q -l smallClassic >/dev/null 2>&1
    python pacman.py -p QLearningAgent -a featureExts=all,db=preTrainRand,save=False -n 1000 -q >rand-pretrain.log 2>&1
    plot rand-pretrain "Pretraining on Smaller Maze, Default Ghost" rand-pretrain.log rand-large.log "Pre-Training" "Fresh"

    rm -f preTrainDir.db
    python pacman.py -p QLearningAgent -a featureExts=all,db=preTrainDir -n 1000 -q -l smallClassic -g DirectionalGhost >/dev/null 2>&1
    python pacman.py -p QLearningAgent -a featureExts=all,db=preTrain,save=False -n 1000 -q -g DirectionalGhost >dir-pretrain.log 2>&1
    plot dir-pretrain "Pretraining on Smaller Maze, Directional Ghost" dir-pretrain.log dir-large.log "Pre-Training" "Fresh"
}

goal1
goal2
