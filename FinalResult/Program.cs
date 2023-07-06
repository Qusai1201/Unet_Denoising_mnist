using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using Models;

#pragma warning disable CA1416



class concatenation
{
    private static void CopyImageData(Bitmap src, int x, int y, BitmapData destData)
    {
        BitmapData srcData = src.LockBits(new Rectangle(0, 0, src.Width, src.Height), ImageLockMode.ReadOnly, PixelFormat.Format8bppIndexed);
        int Width = srcData.Width;
        int Height = srcData.Height;
        int SrcStride = srcData.Stride;
        int DestStride = destData.Stride;

        int destX = x;
        int destY = y;

        unsafe
        {
            byte* srcPtr = (byte*)srcData.Scan0;
            byte* destPtr = (byte*)destData.Scan0;

            for (int i = 0; i < Height; i++)
            {
                int DestRow = (destY + i) * DestStride;
                int SrcRow = i * SrcStride;
                for (int j = 0; j < Width; j++)
                {
                    byte pixel = (byte)(srcPtr[j + SrcRow]);
                    destPtr[DestRow + destX + j] = pixel;
                }
            }
        }
        src.UnlockBits(srcData);
    }
    public static Bitmap concatenate(Bitmap image1, Bitmap image2)
    {

        int Width = Math.Max(image1.Width, image2.Width);
        int Height = image1.Height + image2.Height;



        Bitmap result = new Bitmap(Width, Height, PixelFormat.Format8bppIndexed);

        ColorPalette platte = result.Palette;
        for (int i = 0; i < 256; i++)
        {
            platte.Entries[i] = Color.FromArgb(i, i, i);
        }
        result.Palette = platte;

        BitmapData resultData = result.LockBits(new Rectangle(0, 0, Width, Height), ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);


        CopyImageData(image1, (Width - image1.Width) / 2, 0, resultData);
        CopyImageData(image2, (Width - image2.Width) / 2, image1.Height, resultData);
        return result;

    }
}

class Classify
{
    static void Main(string[] args)
    {
        Model classification = new Model("models/Unet_autoencoder.onnx", "models/CNN_mnist.onnx");


        string[] pathArray = Directory.GetFiles(@"Tests/", "*.png");

        Console.WriteLine("starting classification");

        foreach (string ImgPath in pathArray)
        {

            Console.WriteLine(ImgPath);

            Bitmap src = new Bitmap(ImgPath);

            Bitmap pred = classification.DenoisePredict(src);

            string path = classification.DigitPredict(pred);

            concatenation.concatenate(src, pred).Save("Predictions/" + path + ".png");
        }

    }

}