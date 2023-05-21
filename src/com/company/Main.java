package com.company;

import cern.colt.Arrays;
import com.company.moa.classifiers.meta.imbalanced.ROSE;
import com.company.moa.classifiers.trees.RandomSubspaceHT;
import moa.MOAObject;
import moa.classifiers.Classifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.functions.SGD;
import moa.core.InstanceExample;
import moa.core.ObjectRepository;
import moa.core.SerializeUtils;
import moa.streams.generators.RandomRBFGenerator;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import weka.core.SerializationHelper;

public class Main {

    public static void main(String[] args) throws Exception {
        // Create a classifier
        //Classifier learner = new ROSE();

        Classifier learner = new RandomSubspaceHT();
        // Create a stream
        RandomRBFGenerator stream = new RandomRBFGenerator();
        stream.prepareForUse();

        // Set the model context and prepare the classifier
        learner.setModelContext(stream.getHeader());
        learner.prepareForUse();

        // Train the classifier on the stream
        int numInstances = 10000;
        int numberSamplesCorrect = 0;
        int numberSamples = 0;
        boolean isTesting = true;
        while (stream.hasMoreInstances() && numberSamples < numInstances) {
            InstanceExample trainInst = stream.nextInstance();
            if (isTesting) {
                if (learner.correctlyClassifies(trainInst.instance)) {
                    numberSamplesCorrect++;
                }
            }
            numberSamples++;
            learner.trainOnInstance(trainInst);
            //
            double[] prediction = learner.getVotesForInstance(trainInst);
            System.out.println(trainInst.toString() + "," + Arrays.toString(prediction));
        }
        //MOAObject model= learner.getModel();


        // Print the accuracy of the classifier
        double accuracy = 100.0 * (double) numberSamplesCorrect / (double) numberSamples;
        System.out.println(numberSamples + " instances processed with " + accuracy + "% accuracy");


        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("model.ser"))) {
            oos.writeObject(learner);
        }
        SerializationHelper.write("a.model", learner);
        System.out.println("ok");
    }
}