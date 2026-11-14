using CNTK;
using System.Diagnostics;
using static beadando.BatchTraining;

namespace beadando
{
    class BatchTraining
    {
        readonly DeviceDescriptor device = DeviceDescriptor.CPUDevice;
        readonly DataSet trainDataset;
        readonly DataSet validationDataset;
        readonly Variable x;
        readonly Function y;

        public int HiddenNeuronCount { get; private set; }
        public int HiddenNeuronCount2 { get; private set; }
        public int HiddenNeuronCount3 { get; private set; }

        const int epochCount = 100;

        const int batchSize = 128;

        const int basePatience = 10;


        public BatchTraining(DataSet train_dataset, DataSet validation_dataset, int depth)
        {
            this.trainDataset = train_dataset;
            this.validationDataset = validation_dataset;

            switch (depth)
            {
                case 1:
                    HiddenNeuronCount = 64;
                    x = Variable.InputVariable(new int[] { trainDataset.InputSize }, DataType.Float, "x");
                    
                    Parameter w1 = new Parameter(new int[] { HiddenNeuronCount, trainDataset.InputSize }, DataType.Float, CNTKLib.GlorotNormalInitializer());
                    Parameter b = new Parameter(new int[] { HiddenNeuronCount }, DataType.Float, 0);

                    Function h1 = CNTKLib.ReLU(CNTKLib.Plus(CNTKLib.Times(w1, x), b));

                    Parameter w2 = new Parameter(new int[] { trainDataset.OutputSize, HiddenNeuronCount }, DataType.Float, CNTKLib.GlorotNormalInitializer());
                    Parameter b2 = new Parameter(new int[] { trainDataset.OutputSize }, DataType.Float, 0);

                    y = CNTKLib.Sigmoid(CNTKLib.Plus(CNTKLib.Times(w2, h1), b2));
                    break;

                case 2:
                    HiddenNeuronCount = 128;
                    HiddenNeuronCount2 = 64;

                    x = Variable.InputVariable(new int[] { trainDataset.InputSize }, DataType.Float, "x");

                    w1 = new Parameter(new int[] { HiddenNeuronCount, trainDataset.InputSize }, DataType.Float, CNTKLib.GlorotNormalInitializer());
                    b = new Parameter(new int[] { HiddenNeuronCount }, DataType.Float, 0);

                    h1 = CNTKLib.ReLU(CNTKLib.Plus(CNTKLib.Times(w1, x), b));

                    w2 = new Parameter(new int[] { HiddenNeuronCount2, HiddenNeuronCount }, DataType.Float, CNTKLib.GlorotNormalInitializer());
                    b2 = new Parameter(new int[] { HiddenNeuronCount2 }, DataType.Float, 0);

                    Function h2 = CNTKLib.ReLU(CNTKLib.Plus(CNTKLib.Times(w2, h1), b2));

                    Parameter w3 = new Parameter(new int[] { trainDataset.OutputSize, HiddenNeuronCount2 }, DataType.Float, CNTKLib.GlorotNormalInitializer());
                    Parameter b3 = new Parameter(new int[] { trainDataset.OutputSize }, DataType.Float, 0);

                    y = CNTKLib.Sigmoid(CNTKLib.Plus(CNTKLib.Times(w3, h2), b3));
                    break;

                case 3:
                    HiddenNeuronCount = 256;
                    HiddenNeuronCount2 = 128;
                    HiddenNeuronCount3 = 64;
                    x = Variable.InputVariable(new int[] { trainDataset.InputSize }, DataType.Float, "x");

                    w1 = new Parameter(new int[] { HiddenNeuronCount, trainDataset.InputSize }, DataType.Float, CNTKLib.GlorotNormalInitializer());
                    b = new Parameter(new int[] { HiddenNeuronCount }, DataType.Float, 0);

                    h1 = CNTKLib.ReLU(CNTKLib.Plus(CNTKLib.Times(w1, x), b));

                    w2 = new Parameter(new int[] { HiddenNeuronCount2, HiddenNeuronCount }, DataType.Float, CNTKLib.GlorotNormalInitializer());
                    b2 = new Parameter(new int[] { HiddenNeuronCount2 }, DataType.Float, 0);

                    h2 = CNTKLib.ReLU(CNTKLib.Plus(CNTKLib.Times(w2, h1), b2));

                    w3 = new Parameter(new int[] { HiddenNeuronCount3, HiddenNeuronCount2 }, DataType.Float, CNTKLib.GlorotNormalInitializer());
                    b3 = new Parameter(new int[] { HiddenNeuronCount3 }, DataType.Float, 0);

                    Function h3 = CNTKLib.ReLU(CNTKLib.Plus(CNTKLib.Times(w3, h2), b3));

                    Parameter w4 = new Parameter(new int[] { trainDataset.OutputSize, HiddenNeuronCount3 }, DataType.Float, CNTKLib.GlorotNormalInitializer());
                    Parameter b4 = new Parameter(new int[] { trainDataset.OutputSize }, DataType.Float, 0);

                    y = CNTKLib.Sigmoid(CNTKLib.Plus(CNTKLib.Times(w4, h3), b4));
                    break;

                default:
                    throw new ArgumentException("Depth must be 1/2/3.");


            }
            
        }

        public BatchTraining(string filename)
        {
            y = Function.Load(filename, device);
            x = y.Arguments.Single(x => x.Name == "x");
        }

        public void TrainAndValidate()
        {

            //ki kell egészíteni a gráfot
            Variable yt = Variable.InputVariable(new int[] {trainDataset.OutputSize }, DataType.Float);
            Function loss = CNTKLib.BinaryCrossEntropy(y, yt);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, yt);

            //mivel tanítunk
            Learner learner = CNTKLib.SGDLearner(new ParameterVector(y.Parameters().ToArray()), new TrainingParameterScheduleDouble(0.01, 1));
            Trainer trainer = Trainer.CreateTrainer(y, loss, y_yt_equal, new List<Learner>() { learner });

            double bestValidationLoss = double.MaxValue;

            int patience = basePatience;

            for (int epochI = 1; epochI <= epochCount; epochI++)
            {

                double sumLoss = 0;
                double sumEval = 0;
                int samplesSeen = 0;

                trainDataset.Shuffle();
                for (int start = 0; start < trainDataset.Count; start += batchSize)
                {
                    int currentBatchSize = Math.Min(batchSize, trainDataset.Count - start);

                    int inputStart = start * trainDataset.InputSize;
                    int inputCount = currentBatchSize * trainDataset.InputSize;

                    int outputStart = start * trainDataset.OutputSize;
                    int outputCount = currentBatchSize * trainDataset.OutputSize;

                    Value x_val = Value.CreateBatch(x.Shape, trainDataset.Input.GetRange(inputStart, inputCount), device);
                    Value yt_val = Value.CreateBatch(yt.Shape, trainDataset.Output.GetRange(outputStart, outputCount), device);

                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        {x, x_val },
                        {yt, yt_val }
                    };

                    bool isSweepEnd = start + currentBatchSize >= trainDataset.Count;

                    trainer.TrainMinibatch(inputDataMap, isSweepEnd, device);

                    int samples = (int)trainer.PreviousMinibatchSampleCount();

                    sumLoss += trainer.PreviousMinibatchLossAverage() * samples;
                    sumEval += trainer.PreviousMinibatchEvaluationAverage() * samples;

                    samplesSeen += samples;
                }


                EvaluationResult evaluationResult = new EvaluationResult();

                for (int start = 0; start < validationDataset.Count; start += batchSize)
                {

                    int currentBatchSize = Math.Min(batchSize, validationDataset.Count - start);

                    int inputStart = start * validationDataset.InputSize;

                    int inputCount = currentBatchSize * validationDataset.InputSize;

                    int outputStart = start * validationDataset.OutputSize;

                    int outputCount = currentBatchSize * validationDataset.OutputSize;

                    Value x_val = Value.CreateBatch(x.Shape, validationDataset.Input.GetRange(inputStart, inputCount), device);

                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { x, x_val },
                    };

                    var outputDataMap = new Dictionary<Variable, Value>()
                    {
                        { y, null },
                    };

                    y.Evaluate(inputDataMap, outputDataMap, device);

                    var predictions = outputDataMap[y].GetDenseData<float>(y);

                    for (int i = 0; i < currentBatchSize; i++)
                    {

                        double probability = predictions[i][0];

                        // clamp, hogy ne legyen log(0)
                        probability = Math.Clamp(probability, 1e-7, 1.0 - 1e-7);

                        int predicted = probability >= 0.5 ? 1 : 0;
                        int actual = (int)validationDataset.Output[outputStart + i];

                        evaluationResult.Loss += -(actual * Math.Log(probability) + (1 - actual) * Math.Log(1 - probability));

                        if (predicted == 1 && actual == 1)
                            evaluationResult.TP++;
                        else if (predicted == 0 && actual == 0)
                            evaluationResult.TN++;
                        else if (predicted == 1 && actual == 0)
                            evaluationResult.FP++;
                        else
                            evaluationResult.FN++;
                    }

                }
                int total = evaluationResult.TP + evaluationResult.TN + evaluationResult.FP + evaluationResult.FN;
                evaluationResult.Loss /= total;
                evaluationResult.Accuracy = (double)(evaluationResult.TP + evaluationResult.TN) / total;
                evaluationResult.Precision = (double)evaluationResult.TP / (evaluationResult.TP + evaluationResult.FP);
                evaluationResult.Recall = (double)evaluationResult.TP / (evaluationResult.TP + evaluationResult.FN);
                evaluationResult.F1 = 2 * evaluationResult.Precision * evaluationResult.Recall / (evaluationResult.Precision + evaluationResult.Recall);

                Console.Clear();
                Console.WriteLine(String.Format("{0}\n", epochI));
                Console.WriteLine("TRAINING\n");
                Console.WriteLine(String.Format("train loss:{0}\ttrain eval:{1}", sumLoss / samplesSeen, sumEval / samplesSeen));
                Console.WriteLine();
                Console.WriteLine("VALIDATION\n");
                Console.WriteLine(String.Format("validation loss:{0}", evaluationResult.Loss));
                Console.WriteLine(String.Format("Accuracy: {0}\t Precision: {1}\t Recall: {2}\t F1: {3}", evaluationResult.Accuracy, evaluationResult.Precision, evaluationResult.Recall, evaluationResult.F1));
                Console.WriteLine(String.Format("TP: {0}\t TN: {1}\t FP: {2}\t FN: {3}\n", evaluationResult.TP, evaluationResult.TN, evaluationResult.FP, evaluationResult.FN));

                if (evaluationResult.Loss < bestValidationLoss)
                {
                    y.Save($@"..\..\..\..\bestmodel_{trainDataset.InputSize}_{HiddenNeuronCount}_{(HiddenNeuronCount2 == 0 ? "" :HiddenNeuronCount2.ToString() + '_')}{(HiddenNeuronCount3 == 0 ? "" : HiddenNeuronCount3.ToString() + '_')}{trainDataset.OutputSize}.cntk");

                    bestValidationLoss = evaluationResult.Loss;
                    Console.WriteLine("Model Saved!");
                    patience = basePatience;
                }
                else
                {
                    Console.WriteLine($"Loss did not improve, patience: {--patience}");
                }

                if (patience == 0)
                {
                    Console.WriteLine($"Model hasnt improved in the last {basePatience} epochs, stopping early.");
                    break;
                }
            }
        }

        public EvaluationResult Test(DataSet testDataset)
        {
            EvaluationResult evaluationResult = new EvaluationResult();

            for (int start = 0; start < testDataset.Count; start++)
            {
                Console.WriteLine($"Value {start}, testing...");
                int currentBatchSize = Math.Min(batchSize, testDataset.Count - start);

                int inputStart = start * testDataset.InputSize;

                int inputCount = currentBatchSize * testDataset.InputSize;

                int outputStart = start * testDataset.OutputSize;

                int outputCount = currentBatchSize * testDataset.OutputSize;

                Value x_val = Value.CreateBatch(x.Shape, testDataset.Input.GetRange(inputStart, inputCount), device);

                var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { x, x_val },
                    };

                var outputDataMap = new Dictionary<Variable, Value>()
                    {
                        { y, null },
                    };

                y.Evaluate(inputDataMap, outputDataMap, device);

                var predictions = outputDataMap[y].GetDenseData<float>(y);

                for (int i = 0; i < currentBatchSize; i++)
                {

                    double probability = predictions[i][0];

                    // clamp, hogy ne legyen log(0)
                    probability = Math.Clamp(probability, 1e-7, 1.0 - 1e-7);

                    int predicted = probability >= 0.5 ? 1 : 0;
                    int actual = (int)testDataset.Output[outputStart + i];

                    evaluationResult.Loss += -(actual * Math.Log(probability) + (1 - actual) * Math.Log(1 - probability));

                    if (predicted == 1 && actual == 1)
                        evaluationResult.TP++;
                    else if (predicted == 0 && actual == 0)
                        evaluationResult.TN++;
                    else if (predicted == 1 && actual == 0)
                        evaluationResult.FP++;
                    else
                        evaluationResult.FN++;
                }
                int total = evaluationResult.TP + evaluationResult.TN + evaluationResult.FP + evaluationResult.FN;
                evaluationResult.Loss /= total;
                evaluationResult.Accuracy = (double)(evaluationResult.TP + evaluationResult.TN) / total;
                evaluationResult.Precision = (double)evaluationResult.TP / (evaluationResult.TP + evaluationResult.FP);
                evaluationResult.Recall = (double)evaluationResult.TP / (evaluationResult.TP + evaluationResult.FN);
                evaluationResult.F1 = 2 * evaluationResult.Precision * evaluationResult.Recall / (evaluationResult.Precision + evaluationResult.Recall);
            }

            return evaluationResult;
        }

        public class EvaluationResult
        {
            public double Precision;
            public double Recall;
            public double F1;
            public double Accuracy;
            public double Loss;
            public int TP;
            public int TN;
            public int FP;
            public int FN;
        }
    }
}
