using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

#pragma warning disable CA1416

namespace Models
{

    public class Model
    {

        private InferenceSession DenoiseSession;
        private InferenceSession PredictionSession;


        public Model(string DenoiseModelPath, string PredictionModelPath)
        {
            DenoiseSession = new InferenceSession(DenoiseModelPath);
            PredictionSession = new InferenceSession(PredictionModelPath);

        }
        public Bitmap DenoisePredict(Bitmap img)
        {

            float[] image = PreprocessTestImage(img);
            float[] pixleArray = Denoise(image);

            return MakeImage(pixleArray);

        }

        public string DigitPredict(Bitmap img)
        {

            string Prediction = "";
            for (int i = 0; i <= 9; i++)
            {
                Rectangle rect = new Rectangle(i * 28, 0, 28, 28);
                Bitmap image = img.Clone(rect, img.PixelFormat);
                Prediction += Predict(PreprocessTestImage(image));
            }
            return Prediction;
        }

        private unsafe float[] PreprocessTestImage(Bitmap img)
        {
            int Width = img.Width, Height = img.Height;

            BitmapData GrayData = img.LockBits(new Rectangle(0, 0, Width, Height),
          ImageLockMode.ReadWrite, PixelFormat.Format8bppIndexed);

            int stride = GrayData.Stride;
            byte* Ptr = (byte*)GrayData.Scan0;

            float[] result = new float[Width * Height];

            int ImageLength = Width * Height;

            for (int i = 0; i < ImageLength; i++)
            {
                result[i] = Ptr[i] / 255.0f;
            }
            img.UnlockBits(GrayData);
            return result;
        }
        
        private unsafe Bitmap MakeImage(float[] image)
        {
            Bitmap img = new Bitmap(280, 28, PixelFormat.Format8bppIndexed);

            BitmapData GrayData = img.LockBits(new Rectangle(0, 0, 280, 28),
          ImageLockMode.ReadWrite, PixelFormat.Format8bppIndexed);

            int stride = GrayData.Stride;
            byte* Ptr = (byte*)GrayData.Scan0;


            ColorPalette platte = img.Palette;
            for (int i = 0; i < 256; i++)
            {
                platte.Entries[i] = Color.FromArgb(i, i, i);
            }
            img.Palette = platte;

            int ImageLength = image.Length;

            for (int i = 0; i < ImageLength; i++)
            {
                if (image[i] * 255 > 255)
                    Ptr[i] = 255;
                else
                    Ptr[i] = (byte)(image[i] * 255);
            }
            img.UnlockBits(GrayData);
            return img;
        }

        private float[] Denoise(float[] image)
        {
            var modelInputLayerName = this.DenoiseSession.InputMetadata.Keys.Single();

            int[] dimensions = { 1, 28, 280, 1 };
            var inputTensor = new DenseTensor<float>(image, dimensions);
            var modelInput = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(modelInputLayerName, inputTensor)
                };

            var result = DenoiseSession.Run(modelInput);
            return ((DenseTensor<float>)result.Single().Value).ToArray();
        }

        
        private string Predict(float[] image)
        {
            var modelInputLayerName = this.PredictionSession.InputMetadata.Keys.Single();

            int[] dimensions = { 1, 28, 28, 1 };
            var inputTensor = new DenseTensor<float>(image, dimensions);
            var modelInput = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(modelInputLayerName, inputTensor)
                };

            var result = PredictionSession.Run(modelInput);
            float[] output = ((DenseTensor<float>)result.Single().Value).ToArray();

            float Prediction = output.Max();
            int index = Array.IndexOf(output, Prediction);

            return index.ToString();
        }
    }
}   