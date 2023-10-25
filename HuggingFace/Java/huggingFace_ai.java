package ai.djl.huggingface.tokenizers;

import ai.djl.ModelException;
import ai.djl.huggingface.translator.QuestionAnsweringTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;




 public void testTokenizerDecoding() throws IOException {
        long[][] testIds = {
            {101, 8667, 117, 194, 112, 1155, 106, 1731, 1132, 1128, 100, 136, 102},
            {101, 3570, 1110, 170, 21162, 1285, 119, 2750, 4250, 146, 112, 173, 1474, 102}
        };
        String[] expectedDecodedNoSpecialTokens = {
            "Hello, y ' all! How are you?", "Today is a sunny day. Good weather I ' d say"
        };
        String[] expectedDecodedWithSpecialTokens = {
            "[CLS] Hello, y ' all! How are you [UNK]? [SEP]",
            "[CLS] Today is a sunny day. Good weather I ' d say [SEP]"
        };
        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("bert-base-cased")) {
            for (int i = 0; i < testIds.length; ++i) {
                String ret = tokenizer.decode(testIds[i]);
                Assert.assertEquals(ret, expectedDecodedWithSpecialTokens[i]);
                ret = tokenizer.decode(testIds[i], true);
                Assert.assertEquals(ret, expectedDecodedNoSpecialTokens[i]);
            }
        }


    }

public final class QuestionAnswering {

    private QuestionAnswering() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        String question = "When did BBC Japan start broadcasting?";
        String paragraph =
                "BBC Japan was a general entertainment Channel. "
                        + "Which operated between December 2004 and April 2006. "
                        + "It ceased operations after its Japanese distributor folded.";

        Criteria<QAInput, String> criteria =
                Criteria.builder()
                        .setTypes(QAInput.class, String.class)
                        .optModelUrls(
                                "djl://ai.djl.huggingface.pytorch/deepset/minilm-uncased-squad2")
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new QuestionAnsweringTranslatorFactory())
                        .optProgress(new ProgressBar())
                        .build();

        try (HuggingFaceTokenizer tokenizer = 
                HuggingFaceTokenizer.builder()
                        .optTokenizerName("bert-base-cased")
                        .optAddSpecialTokens(false)
                        .build()) {
            for (int i = 0; i < testIds.length; ++i) {
                String ret = tokenizer.decode(testIds[i]);
                Assert.assertEquals(ret, expectedDecodedNoSpecialTokens[i]);
                ret = tokenizer.decode(testIds[i], false);
                Assert.assertEquals(ret, expectedDecodedWithSpecialTokens[i]);
            }
        }

        tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(path));
        try (ZooModel<QAInput, String> model = criteria.loadModel();
                Predictor<QAInput, String> predictor = model.newPredictor()) {
            QAInput input = new QAInput(question, paragraph);
            String res = predictor.predict(input);
            System.out.println("answer: " + res);
        }
    }
}