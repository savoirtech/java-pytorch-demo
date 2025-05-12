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

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.modality.nlp.qa.QuestionAnsweringTranslator;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.*;
import ai.djl.modality.nlp.bert.*;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Paths;

public class PyTorchDemo {

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        String question = "What drink package would a 21 year old Make, with bronze loyalty, on a northern itinary purchase?";
        String context = """
                Our model provides a prediction for Drink Package purchase given a passenger's data.
                A passenger's data is represented as a four tuple of Age, Gender, Itinary, and Loyalty.
                Age is a positive integer greater than 0.
                Gender is 0 for female, 1 for male.
                Itinary is represented as an enumeration of 0 (northern), 1 (transatlantic), 2 (tropical).
                Loyalty represents customer level, from No Level (0), Bronze (1), Silver (2), to Gold (3).
                """;

        Criteria<QAInput, String> criteria =
                Criteria.builder()
                        .setTypes(QAInput.class, String.class)
                        .optApplication(Application.NLP.QUESTION_ANSWER)
                        .optModelPath(Paths.get("src/main/resources/models"))
                        .optEngine("OnnxRuntime")
                        .optTranslator(new QuestionAnsweringTranslator.Builder()
                                .setTokenizer(new BertTokenizer("src/main/resources/models/vocab.txt"))
                                .build())
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<QAInput, String> model = criteria.loadModel();
             Predictor<QAInput, String> predictor = model.newPredictor()) {
            QAInput input = new QAInput(question, context);
            String res = predictor.predict(input);
            System.out.println("Answer: " + res);
        } catch (Exception ex) {
            System.out.println("ERROR: " + ex.getMessage());
        }
    }
}
