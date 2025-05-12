/*
 * Copyright (c) 2012-2025 Savoir Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.savoir.java.pytorch.demo;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.ndarray.types.Shape;
import java.io.IOException;
import java.nio.file.Paths;

public class OnnxDemo {

    public static void main(String[] args) throws IOException, ModelException, TranslateException {

        Criteria<NDList, NDList> criteria = Criteria.builder()
                .setTypes(NDList.class, NDList.class) // Input and output types
                        .optModelPath(Paths.get("src/main/resources/models"))
                        .optModelName("cruise-drink-package-recommender.onnx")
                        .optEngine("OnnxRuntime")
                        .optProgress(new ProgressBar())
                        .build();

        // Step 2: Load the model
        ZooModel<NDList, NDList> model = criteria.loadModel();

        // Step 3: Prepare input data as NDList (e.g., a single input tensor)
        NDManager manager = NDManager.newBaseManager();
        //Age 21, Male, Northern, Bronze Loyalty
        NDArray inputTensor = manager.create(new float[]{21, 1, 0, 1}, new Shape(1, 4));
        NDList input = new NDList(inputTensor);

        // Step 4: Use the predictor to perform inference
        try (Predictor<NDList, NDList> predictor = model.newPredictor()) {
            // Perform prediction with input NDList
            NDList result = predictor.predict(input);

            // Step 5: Print the output
            System.out.println("Model Prediction Output:" + result.get(0));
        } catch (Exception ex) {
            System.out.println("ERROR: " + ex.getMessage());
        }
    }
}
