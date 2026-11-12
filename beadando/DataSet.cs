using System;
using System.Collections.Generic;
using System.Text;

namespace beadando
{
    public class DataSet
    {
        public readonly int InputSize = 70;
        public List<float> Input { get; set; } = new List<float>();

        public readonly int OutputSize = 1;
        public List<float> Output { get; set; } = new List<float>();

        public int Count { get; set; }

        public DataSet(string filename)
        {
            LoadData(filename);
        }

        void LoadData(string filename)
        {
            Count = 0;

            foreach (String line in File.ReadAllLines(filename))
            {
                var floats = line.Split(',').Select(x => float.Parse(x)).ToList();
                //add features to input except label
                Input.AddRange(floats.GetRange(0, InputSize));
                //add label to output
                Output.Add(floats[InputSize]);
                Count++;
            }
        }

        public void Shuffle()
        {
            Random rnd = new Random();

            for (int swapI = 0; swapI < Count; swapI++)
            {
                var a = rnd.Next(Count);
                var b = rnd.Next(Count);

                if (a != b)
                {
                    float T;
                    for (int i = 0; i < InputSize; i++)
                    {
                        T = Input[a * InputSize + i];
                        Input[a * InputSize + i] = Input[b * InputSize + i];
                        Input[b * InputSize + i] = T;
                    }
                    T = Output[a];
                    Output[a] = Output[b];
                    Output[b] = T;
                }
            }
        }

    }
}
