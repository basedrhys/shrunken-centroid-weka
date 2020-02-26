package weka.classifiers.lazy;

import weka.core.DenseInstance;
import weka.core.Instance;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

class Centroid implements Serializable  {
    private static final long serialVersionUID = 0;

    // Instance to hold the values
    private Instance m_inst;

    // Instance to hold the values of the shrunken version
    private Instance m_shrunkenInst;

    // Keep a list of instances for this class
    private List<Instance> m_instances;

    private int m_classIndex;

    public Centroid(int numAttributes, int classAttribute) {
        m_inst = new DenseInstance(numAttributes, new double[numAttributes]);
        m_shrunkenInst = new DenseInstance(numAttributes, new double[numAttributes]);
        m_instances = new ArrayList<>();
        m_classIndex = classAttribute;
    }

    public void setValue(int i, double v) {
        m_inst.setValue(i, v);
    }

    public void setShrunkenValue(int i, double v) { m_shrunkenInst.setValue(i, v);}

    public double getValue(int i) {
        return m_inst.value(i);
    }

    public int getNonZeroShrunkenAttributes(Centroid globalCentroid) {
        int count = 0;
        for (int i = 0; i < m_shrunkenInst.numAttributes(); i++) {
            if (i != m_classIndex) {
                double val = Math.abs(m_shrunkenInst.value(i) - globalCentroid.getValue(i));
                if (val > 0) {
                    count++;
                }
            }
        }
        return count;
    }

    public void addInstance(Instance inst) {
        // Add all attribute values from this instance to both the global centroid
        // and the appropriate class centroid
        for (int i = 0; i < m_inst.numAttributes(); i++) {
            if (i != m_classIndex) {
                double newVal = m_inst.value(i) + inst.value(i);
                m_inst.setValue(i, newVal);
            }
        }
        m_instances.add(inst);
    }

    public Instance getInstance() {
        return m_inst;
    }

    public List<Instance> getInstances() {
        return m_instances;
    }

    public int numAttributes() { return m_inst.numAttributes(); }

    public double getDifferenceFromInstanceAttribute(Instance instance, int attributeI) {
        return instance.value(attributeI) - m_inst.value(attributeI);
    }

    public double getDifferenceFromInstanceAttribute(Instance instance, int attributeI, boolean useShrunken) {
        if (useShrunken)
            return instance.value(attributeI) - m_shrunkenInst.value(attributeI);
        else
            return getDifferenceFromInstanceAttribute(instance, attributeI);
    }

    public double getDistanceFromInstance(Instance instance) {
        double dist = 0;
        for (int i = 0; i < m_inst.numAttributes(); i++) {
            if (i != m_classIndex){
                double d = instance.value(i) - m_inst.value(i);
                dist += d * d;
            }
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