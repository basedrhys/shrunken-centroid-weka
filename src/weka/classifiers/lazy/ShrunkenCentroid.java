package weka.classifiers.lazy;

import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

import java.util.HashMap;
import java.util.Map;

public class ShrunkenCentroid extends AbstractClassifier {

    protected int m_K = 1;

    protected NearestNeighbourSearch m_NNSearch = new LinearNNSearch();

    @OptionMetadata(displayName = "number of neighbours", description = "Number of neighbours to use (default = 1).",
            commandLineParamName = "K", commandLineParamSynopsis = "-K <int>", displayOrder = 1)
    public void setK(int k) {
        m_K = k;
    }

    public int getK() {
        return m_K;
    }

    private Centroid m_globalCentroid;

    private Map<String, Centroid> m_classCentroids;

    public void buildClassifier(Instances trainingData) throws Exception {
        trainingData = new Instances(trainingData);
        trainingData.deleteWithMissingClass();

        createCentroids(trainingData);



        m_NNSearch.setInstances(trainingData);
    }

    private void printClassCentroids() {
        for (String key : m_classCentroids.keySet()) {
            System.out.println("CLASS = " + key + ", " + m_classCentroids.get(key));
        }
    }

    private void createSI() {
        int i = 0;
    }

    private void createCentroids(Instances trainingData) {
        int centroidNumAttributes = trainingData.numAttributes() - 1; // ignore class value

        m_globalCentroid = new Centroid(centroidNumAttributes);
        m_classCentroids = new HashMap<>();

        // For each instance, add them into the global and class centroid
        for (Instance instance : trainingData) {
            // Get the class value
            String classVal = instance.stringValue(instance.classIndex());
            // Create the centroid for this class if we haven't already
            if (!m_classCentroids.containsKey(classVal)) {
                m_classCentroids.put(classVal, new Centroid(centroidNumAttributes));
            }

            m_globalCentroid.addInstance(instance);

            // Get the centroid for this class
            Centroid tempClassCentroid = m_classCentroids.get(classVal);
            // Add the value for this instance
            tempClassCentroid.addInstance(instance);
            // Save the centroid
            m_classCentroids.put(classVal, tempClassCentroid);
        }

        // After adding up all the attribute values, we finally average them to find the global and class centroid
        m_globalCentroid.averageValues();
        for (String classVal : m_classCentroids.keySet()) {
            Centroid c = m_classCentroids.get(classVal);
            c.averageValues();
            m_classCentroids.put(classVal, c);
        }
    }

    public double[] distributionForInstance(Instance testInstance) throws Exception {

        m_NNSearch.addInstanceInfo(testInstance);

        Instances neighbours = m_NNSearch.kNearestNeighbours(testInstance, m_K);

        double[] dist = new double[testInstance.numClasses()];
        for (Instance neighbour : neighbours) {
            if (testInstance.classAttribute().isNominal()) {
                dist[(int)neighbour.classValue()] += 1.0 / neighbours.numInstances();
            } else {
                dist[0] += neighbour.classValue() / neighbours.numInstances();
            }
        }
        return dist;
    }
}
