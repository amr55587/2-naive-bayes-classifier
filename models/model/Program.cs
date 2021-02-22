using System;
using M = Microsoft.ML.Probabilistic.Models;
using D = Microsoft.ML.Probabilistic.Distributions;
using A = Microsoft.ML.Probabilistic.Algorithms;
using Range = Microsoft.ML.Probabilistic.Models.Range;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Linq;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic;
using System.Runtime.Serialization.Formatters.Binary;
namespace model
{
    class Program
    {
        static void Main(string[] args)
        {
            /********* arguments **********/
            string dataDir = args[0];
            string datasetFilename = args[1];
            /******************************/

            /************ data ************/
            string[] lines = File.ReadAllLines(dataDir+datasetFilename);
            int numSamples = lines.Length;

            double[] x0Data = new double[numSamples];
            double[] x1Data = new double[numSamples];
            bool[] yData = new bool[numSamples];

            for (int i = 0; i < numSamples; i++)
            {
                string[] strArray = lines[i].Split('|');
                double[] doubleArray = Array.ConvertAll<string, double>(strArray, Convert.ToDouble);
                
                x0Data[i] = doubleArray[0];
                x1Data[i] = doubleArray[1];

                if (doubleArray[2] == 1)
                {
                    yData[i] = true;
                }
                else
                {
                    yData[i] = false;
                }
            }
            /********************************/

            /********* model setup **********/
            Range n = new Range(numSamples);

            M.Variable<bool> y = M.Variable.New<bool>();
            M.Variable<double> x0 = M.Variable.New<double>();
            M.Variable<double> x1 = M.Variable.New<double>();

            double variance = 1.0;

            M.Variable<double> x0c0Mean = M.Variable.GaussianFromMeanAndVariance(0, 10);
            M.Variable<double> x0c1Mean = M.Variable.GaussianFromMeanAndVariance(0, 10);
            M.Variable<double> x1c0Mean = M.Variable.GaussianFromMeanAndVariance(0, 10);
            M.Variable<double> x1c1Mean = M.Variable.GaussianFromMeanAndVariance(0, 10);

            // stores the product of all messages sent the means by the previous batches 
            M.Variable<D.Gaussian> x0c0MeanMessage = M.Variable.Observed<D.Gaussian>(D.Gaussian.Uniform());
            M.Variable<D.Gaussian> x0c1MeanMessage = M.Variable.Observed<D.Gaussian>(D.Gaussian.Uniform());
            M.Variable<D.Gaussian> x1c0MeanMessage = M.Variable.Observed<D.Gaussian>(D.Gaussian.Uniform());
            M.Variable<D.Gaussian> x1c1MeanMessage = M.Variable.Observed<D.Gaussian>(D.Gaussian.Uniform());

            M.Variable.ConstrainEqualRandom(x0c0Mean, x0c0MeanMessage);
            M.Variable.ConstrainEqualRandom(x0c1Mean, x0c1MeanMessage);
            M.Variable.ConstrainEqualRandom(x1c0Mean, x1c0MeanMessage);
            M.Variable.ConstrainEqualRandom(x1c1Mean, x1c1MeanMessage);

            x0c0Mean.AddAttribute(QueryTypes.Marginal);
            x0c0Mean.AddAttribute(QueryTypes.MarginalDividedByPrior);
            x0c1Mean.AddAttribute(QueryTypes.Marginal);
            x0c1Mean.AddAttribute(QueryTypes.MarginalDividedByPrior);
            x1c0Mean.AddAttribute(QueryTypes.Marginal);
            x1c0Mean.AddAttribute(QueryTypes.MarginalDividedByPrior);
            x1c1Mean.AddAttribute(QueryTypes.Marginal);
            x1c1Mean.AddAttribute(QueryTypes.MarginalDividedByPrior);

            D.Gaussian x0c0MeanMarginal = D.Gaussian.Uniform();
            D.Gaussian x0c1MeanMarginal = D.Gaussian.Uniform();
            D.Gaussian x1c0MeanMarginal = D.Gaussian.Uniform();
            D.Gaussian x1c1MeanMarginal = D.Gaussian.Uniform();

            M.Variable<double> cPrior = M.Variable.Beta(1, 1);

            y = M.Variable.Bernoulli(cPrior);

            using (M.Variable.IfNot(y))
            {
                x0.SetTo(M.Variable.GaussianFromMeanAndVariance(x0c0Mean, variance));
                x1.SetTo(M.Variable.GaussianFromMeanAndVariance(x1c0Mean, variance));
            }
            using (M.Variable.If(y))
            {
                x0.SetTo(M.Variable.GaussianFromMeanAndVariance(x0c1Mean, variance));
                x1.SetTo(M.Variable.GaussianFromMeanAndVariance(x1c1Mean, variance));
            }

            /******* inference engine *******/
            M.InferenceEngine engine = new M.InferenceEngine(new A.ExpectationPropagation());
            engine.ShowProgress = false;
            // engine.ShowFactorGraph = true;
            /********************************/

            // the less this is, the more important role the prior over the mean is contributing to the posterior.
            double k = 10.0;

            double[] x0c0Meannatural = { 0, 0 };
            double[] x0c1Meannatural = { 0, 0 };
            double[] x1c0Meannatural = { 0, 0 };
            double[] x1c1Meannatural = { 0, 0 };

            var results = new StringBuilder();
            results.AppendLine("classPost|meanPost0|meanPost1|meanPost2|meanPost3");

            for (int t = 0; t < numSamples; t++)
            {
                x0c0MeanMessage.ObservedValue = D.Gaussian.Uniform();
                x0c1MeanMessage.ObservedValue = D.Gaussian.Uniform();
                x1c0MeanMessage.ObservedValue = D.Gaussian.Uniform();
                x1c1MeanMessage.ObservedValue = D.Gaussian.Uniform();

                x0.ObservedValue = x0Data[t];
                x1.ObservedValue = x1Data[t];
                y.ObservedValue = yData[t];

                D.Gaussian x0c0MeanDataLikelihood = engine.Infer<D.Gaussian>(x0c0Mean, QueryTypes.MarginalDividedByPrior);
                D.Gaussian x0c1MeanDataLikelihood = engine.Infer<D.Gaussian>(x0c1Mean, QueryTypes.MarginalDividedByPrior);
                D.Gaussian x1c0MeanDataLikelihood = engine.Infer<D.Gaussian>(x1c0Mean, QueryTypes.MarginalDividedByPrior);
                D.Gaussian x1c1MeanDataLikelihood = engine.Infer<D.Gaussian>(x1c1Mean, QueryTypes.MarginalDividedByPrior);
            
                D.Beta postClass = engine.Infer<D.Beta>(cPrior);

                double x0c0Meanmb, x0c0Meanb;
                x0c0MeanDataLikelihood.GetNatural(out x0c0Meanmb, out x0c0Meanb);
                double x0c1Meanmb, x0c1Meanb;
                x0c1MeanDataLikelihood.GetNatural(out x0c1Meanmb, out x0c1Meanb);
                double x1c0Meanmb, x1c0Meanb;
                x1c0MeanDataLikelihood.GetNatural(out x1c0Meanmb, out x1c0Meanb);
                double x1c1Meanmb, x1c1Meanb;
                x1c1MeanDataLikelihood.GetNatural(out x1c1Meanmb, out x1c1Meanb);

               if (t > k)
               {
                   x0c0Meannatural[0] = (x0c0Meannatural[0] + x0c0Meanmb) * (k / (k + 1));
                   x0c0Meannatural[1] = (x0c0Meannatural[1] + x0c0Meanb) * (k / (k + 1));
                   x0c1Meannatural[0] = (x0c1Meannatural[0] + x0c1Meanmb) * (k / (k + 1));
                   x0c1Meannatural[1] = (x0c1Meannatural[1] + x0c1Meanb) * (k / (k + 1));
                   x1c0Meannatural[0] = (x1c0Meannatural[0] + x1c0Meanmb) * (k / (k + 1));
                   x1c0Meannatural[1] = (x1c0Meannatural[1] + x1c0Meanb) * (k / (k + 1));
                   x1c1Meannatural[0] = (x1c1Meannatural[0] + x1c1Meanmb) * (k / (k + 1));
                   x1c1Meannatural[1] = (x1c1Meannatural[1] + x1c1Meanb) * (k / (k + 1));
               }
               else
               {
                   x0c0Meannatural[0] = x0c0Meannatural[0] + x0c0Meanmb;
                   x0c0Meannatural[1] = x0c0Meannatural[1] + x0c0Meanb;
                   x0c1Meannatural[0] = x0c1Meannatural[0] + x0c1Meanmb;
                   x0c1Meannatural[1] = x0c1Meannatural[1] + x0c1Meanb;
                   x1c0Meannatural[0] = x1c0Meannatural[0] + x1c0Meanmb;
                   x1c0Meannatural[1] = x1c0Meannatural[1] + x1c0Meanb;
                   x1c1Meannatural[0] = x1c1Meannatural[0] + x1c1Meanmb;
                   x1c1Meannatural[1] = x1c1Meannatural[1] + x1c1Meanb;
               }

               x0c0MeanMessage.ObservedValue = new D.Gaussian(x0c0Meannatural[0] / x0c0Meannatural[1], 1 / x0c0Meannatural[1]);
               x0c1MeanMessage.ObservedValue = new D.Gaussian(x0c1Meannatural[0] / x0c1Meannatural[1], 1 / x0c1Meannatural[1]);
               x1c0MeanMessage.ObservedValue = new D.Gaussian(x1c0Meannatural[0] / x1c0Meannatural[1], 1 / x1c0Meannatural[1]);
               x1c1MeanMessage.ObservedValue = new D.Gaussian(x1c1Meannatural[0] / x1c1Meannatural[1], 1 / x1c1Meannatural[1]);
 
               // these are the posterior distribution over the means
               x0c0MeanMarginal = engine.Infer<D.Gaussian>(x0c0Mean);
               x0c1MeanMarginal = engine.Infer<D.Gaussian>(x0c1Mean);
               x1c0MeanMarginal = engine.Infer<D.Gaussian>(x1c0Mean);
               x1c1MeanMarginal = engine.Infer<D.Gaussian>(x1c1Mean);

               var newLine = string.Format("{0}|{1}|{2}|{3}|{4}", postClass.GetMean(), x0c0MeanMarginal.GetMean(), x1c0MeanMarginal.GetMean(), x0c1MeanMarginal.GetMean(), x1c1MeanMarginal.GetMean());
               results.AppendLine(newLine);
            }
            
            File.WriteAllText(dataDir + "results.csv", results.ToString());











            // using (M.Variable.ForEach(n))
            // {
            //     y[n] = M.Variable.Bernoulli(cPrior);

            //     using (M.Variable.IfNot(y[n]))
            //     {
            //         x0[n].SetTo(M.Variable.GaussianFromMeanAndVariance(x0c0Mean, variance));
            //         x1[n].SetTo(M.Variable.GaussianFromMeanAndVariance(x1c0Mean, variance));
            //     }
            //     using (M.Variable.If(y[n]))
            //     {
            //         x0[n].SetTo(M.Variable.GaussianFromMeanAndVariance(x0c1Mean, variance));
            //         x1[n].SetTo(M.Variable.GaussianFromMeanAndVariance(x1c1Mean, variance));
            //     }
            // }
            // /********************************/

            // /********* observations *********/
            // x0.ObservedValue = x0Data;
            // x1.ObservedValue = x1Data;
            // y.ObservedValue = yData;
            // /********************************/

            // /******* inference engine *******/
            // M.InferenceEngine engine = new M.InferenceEngine(new A.ExpectationPropagation());
            // engine.ShowProgress = false;
            // // engine.ShowFactorGraph = true;
            // /********************************/

            // /********** posteriors **********/
            // D.Gaussian postx0c0Mean = engine.Infer<D.Gaussian>(x0c0Mean);
            // D.Gaussian postx0c1Mean = engine.Infer<D.Gaussian>(x0c1Mean);
            // D.Gaussian postx1c0Mean = engine.Infer<D.Gaussian>(x1c0Mean);
            // D.Gaussian postx1c1Mean = engine.Infer<D.Gaussian>(x1c1Mean);
            // D.Beta postClass = engine.Infer<D.Beta>(cPrior);
            // /********************************/

            // /********** print outs **********/
            // Console.WriteLine("Posterior class: {0}", postClass);
            // Console.WriteLine("Posterior class0 means: {0} {1}", postx0c0Mean, postx1c0Mean);
            // Console.WriteLine("Posterior class1 means: {0} {1}", postx0c1Mean, postx1c1Mean);
            // /********************************/

            // /***** creating results.csv *****/
            // var results = new StringBuilder();
            // results.AppendLine("classPost|meanPost0|meanPost1");
            // var line = string.Format("{0}|{1}|{2}", 1-postClass.GetMean(), postx0c0Mean.GetMean(), postx1c0Mean.GetMean());
            // results.AppendLine(line.Replace(',', '.'));
            // line = string.Format("{0}|{1}|{2}", postClass.GetMean(), postx0c1Mean.GetMean(), postx1c1Mean.GetMean());
            // results.AppendLine(line.Replace(',', '.'));
            // File.WriteAllText(dataDir + "results.csv", results.ToString());
            // /*********************************/
        }
    }
}
