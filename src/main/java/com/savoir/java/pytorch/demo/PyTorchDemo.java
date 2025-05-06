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
import ai.djl.huggingface.translator.QuestionAnsweringTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;

public class PyTorchDemo {

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        String question = "Does DJL integrate Java with PyTorch?";
        String paragraph =
                "The Deep Java Library (DJL) is a high-level, " +
                "engine-agnostic framework for deep learning in Java. " +
                "It provides APIs to train and deploy models without " +
                "requiring deep knowledge of specific deep learning engines. " +
                "PyTorch is a popular open-source machine learning framework, " +
                "known for its flexibility and ease of use, particularly in " +
                "research and prototyping. DJL supports PyTorch as one of its backend " +
                "engines, allowing Java developers to leverage PyTorch models " +
                "within their applications.";

        Criteria<QAInput, String> criteria =
                Criteria.builder()
                        .setTypes(QAInput.class, String.class)
                        .optModelUrls("djl://ai.djl.huggingface.pytorch/deepset/minilm-uncased-squad2")
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new QuestionAnsweringTranslatorFactory())
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<QAInput, String> model = criteria.loadModel();
             Predictor<QAInput, String> predictor = model.newPredictor()) {
            QAInput input = new QAInput(question, paragraph);
            String res = predictor.predict(input);
            System.out.println("Answer: " + res);
        } catch (Exception ex) {
            System.err.println(ex.getMessage());
        }
    }
}
