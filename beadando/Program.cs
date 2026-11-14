using CNTK;
using System.Text.Json;

namespace beadando
{
    internal class Program
    {
        static void Main(string[] args)
        {
            DeviceDescriptor.TrySetDefaultDevice(DeviceDescriptor.CPUDevice);

            Console.WriteLine("Press [1] to do a full training + testing workflow, or press [2] for just testing (requires all 3 saved models)");
            var key = Console.ReadKey();

            if (key.Key == ConsoleKey.D1)
            {
                TrainOnly();

                TestOnly();
            }

            if (key.Key == ConsoleKey.D2)
            {
                TestOnly();
            }

            Console.WriteLine("Run finished.");

            Console.ReadLine();
        }

        static void TestOnly()
        {
            DataSet testdata = new DataSet(@"..\..\..\..\test.csv");

            Dictionary<string, BatchTraining> models = new()
            {
                { "70-64-1", new BatchTraining(@"..\..\..\..\bestmodel_70_64_1.cntk") },
                { "70-128-64-1", new BatchTraining(@"..\..\..\..\bestmodel_70_128_64_1.cntk") },
                { "70-256-128-64-1", new BatchTraining(@"..\..\..\..\bestmodel_70_256_128_64_1.cntk") }
            };

            Dictionary<string, BatchTraining.EvaluationResult> results = new();

            foreach (var model in models)
            {
                results.Add(
                    model.Key,
                    model.Value.Test(testdata)
                );
            }

            Console.Clear();

            foreach (var result in results)
            {
                Console.WriteLine($"MODEL: {result.Key}");
                Console.WriteLine($"Loss:      {result.Value.Loss}");
                Console.WriteLine($"Accuracy:  {result.Value.Accuracy}");
                Console.WriteLine($"Precision: {result.Value.Precision}");
                Console.WriteLine($"Recall:    {result.Value.Recall}");
                Console.WriteLine($"F1:        {result.Value.F1}");
                Console.WriteLine();

                Console.WriteLine(
                    $"TP: {result.Value.TP} " +
                    $"TN: {result.Value.TN} " +
                    $"FP: {result.Value.FP} " +
                    $"FN: {result.Value.FN}"
                );

                Console.WriteLine();
            }

            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                IncludeFields = true
            };

            string json = JsonSerializer.Serialize(results, options);

            File.WriteAllText(
                @"..\..\..\..\results.json",
                json
            );

            Console.WriteLine("Saved results to results.json");
        }

        static void TrainOnly()
        {
            DataSet traindata = new DataSet(@"..\..\..\..\train.csv");
            DataSet validationdata = new DataSet(@"..\..\..\..\validate.csv");

            BatchTraining app;

            Console.WriteLine("Beginning training...");
            Thread.Sleep(3000);

            for (int i = 1; i < 4; i++)
            {
                Console.Clear();
                Console.WriteLine($"Training model with network depth {i}");
                app = new BatchTraining(traindata, validationdata, i);
                app.TrainAndValidate();

            }

            Console.WriteLine("Training complete, starting tests...");
            Thread.Sleep(3000);
        }
    }
}
