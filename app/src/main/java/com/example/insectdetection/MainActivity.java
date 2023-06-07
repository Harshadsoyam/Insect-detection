package com.example.insectdetection;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.insectdetection.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity<pestStr> extends AppCompatActivity {

    private ImageView imgView;
    private Button select ,predict;
    private TextView tv;
    private Bitmap img;
    String[] str={"Name : Aphides                 Insecticide : Tata Sentry                Natures Plus Organic             Surya Organic Neem Oil",
            "Name : Armyworm                Insecticide : Bayer Jump                 Tata Tafgor                      Plantic 3 In 1 Fungicide",
            "Name : Beetle                  Insecticide : Hindol                     Bayer Fenos                      Syngenta ekalux",
            "Name : Bollworm                Insecticide : F16                        Gharda Hamla                     Proclaim Insecticide",
            "Name : Mite                    Insecticide : Bayer Oberon               Natures Plus Organic             Dhanuka Omite",
            "Name : Sawfly                  Insecticide : Green Dragon               Tata Sentry                      Surya Organic Neem Oil",
            "Name : Stem-Borer              Insecticide : Tata Rallis                Dhanuka Mortar                   Bayer Premise SC"};

    public int getIndexOfLargest( float[] array )
    {
        if ( array == null || 6== 0 ) return -1; // null or empty

        int largest = 0;
        for ( int i = 1; i < 6; i++ )
        {
            if ( array[i] > array[largest] ) largest = i;
        }
        return largest; // position of the first largest found
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imgView=(ImageView) findViewById(R.id.imageView);
        tv=(TextView) findViewById(R.id.textView);
        select=(Button) findViewById(R.id.button);
        predict=(Button)findViewById(R.id.button2);


        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent=new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent,100);
            }
        });

        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                img=Bitmap.createScaledBitmap(img,224,224,true);
                try {
                    Model model = Model.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.UINT8);

                    TensorImage tensorImage=new TensorImage(DataType.UINT8);
                    tensorImage.load(img);
                    ByteBuffer byteBuffer=tensorImage.getBuffer();

                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    Model.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    // Releases model resources if no longer used.
                    model.close();

                    int idx=getIndexOfLargest(outputFeature0.getFloatArray());
                    tv.setText(str[idx]);

                } catch (IOException e) {
                    // TODO Handle the exception
                }


            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode==100)
        {
            imgView.setImageURI(data.getData());

            Uri uri=data.getData();
            try {
                img= MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}