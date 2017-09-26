package com.wataru.pupiltracking3_w;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.View;

/**
 * Created by wataru on 2017/09/23.
 */

public class onDrawView extends View {
    double x, y, r;
    int work, row, col, height, width;

    public void tran(double x, double y, int r, int work) {
        this.x = x;
        this.y = y;
        this.r = r;
        this.work = work;
    }

    public void tran2(double x, double y, int row, int col, int height,
                      int width, int work) {
        this.x = x;
        this.y = y;
        this.row = row;
        this.col = col;
        this.height = height;
        this.width = width;
        this.work = work;
    }

    public onDrawView(Context context) {
        super(context);
    }

    @Override
    public void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        Paint p = new Paint();
        p.setStrokeWidth(5);
        p.setColor(Color.BLACK);

        Paint rec = new Paint();
        rec.setStrokeWidth(5);
        rec.setColor(Color.RED);
        switch (work) {
            case 1:
                canvas.drawCircle((float) x, (float) y, (float) r, rec);
                break;
            case 2:
                for (int i = 0; i < row - 1; i++) {
                    canvas.drawLine(width * (i + 1) / row, 0,
                            width * (i + 1) / row, height, p);
                }
                for (int i = 0; i < col - 1; i++) {
                    canvas.drawLine(0, height * (i + 1) / col, width, height
                            * (i + 1) / col, p);
                }
                int i,j;
                if (x < 0) {
                    if (y < 0) {
                        i = 0;
                        j = 0;
                    } else {
                        if (y <= height) {
                            i = 0;
                            j = (int) (y * col / height);
                        } else {
                            i = 0;
                            j = col;
                        }

                    }
                } else {
                    if (x <= width) {
                        i = (int) (x * row / width);
                        if (y < 0) {
                            j = 0;
                        } else {
                            if (y <= height) {
                                j = (int) (y * col / height);
                            } else {
                                j = col;
                            }
                        }
                    } else {
                        i = row;
                        if (y < 0) {
                            j = 0;
                        } else {
                            if (y <= height) {
                                j = (int) (y * col / height);
                            } else {
                                j = col;
                            }
                        }
                    }

                }
                canvas.drawLine(width * (i) / row, height * (j) / col, width * (i + 1) / row, height * (j + 1) / col, rec);
                canvas.drawLine(width * (i) / row, height * (j + 1) / col, width * (i + 1) / row, height * j / col, rec);

                break;
        }
    }
}