package weka.classifiers.lazy;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;

import java.util.ArrayList;
import java.util.List;

class Centroid {
    // Instance to hold the values TODO should we use a double[] instead of Instance?
    Instance m_inst;

    // Keep a list of instances for this class
    List<Instance> m_instances;

    public Centroid(int numAttributes) {
        m_inst = new DenseInstance(numAttributes, new double[numAttributes]);
        m_instances = new ArrayList<>();
    }

    public void addInstance(Instance inst) { // TODO work around for class value not being the last val
        // Add all attribute values from this instance to both the global centroid
        // and the appropriate class centroid
        for (int i = 0; i < m_inst.numAttributes(); i++) {
            double newVal = m_inst.value(i) + inst.value(i);
            m_inst.setValue(i, newVal);
        }
        m_instances.add(inst);
    }

    public Instance getInstance() {
        return m_inst;
    }

    public List<Instance> getInstances() {
        return m_instances;
    }

    public double getDifferenceFromInstanceAttribute(Instance instance, int attributeI) {
        return instance.value(attributeI) - m_inst.value(attributeI);
    }

    public double getDistanceFromInstance(Instance instance) {
        double dist = 0;
        for (int i = 0; i < m_inst.numAttributes(); i++) {
            double d = instance.value(i) - m_inst.value(i);
            dist += d * d;
        }
        return Math.sqrt(dist);
    }

    public void averageValues() {
        // Average all attribute values
        for (int i = 0; i < m_inst.numAttributes(); i++) {
            double newVal = m_inst.value(i) / m_instances.size();
            m_inst.setValue(i, newVal);
        }
    }

    public String toString() {
        return String.format("Center = %s, Number of Instances = %d", m_inst.toString(), m_instances.size());
    }

}