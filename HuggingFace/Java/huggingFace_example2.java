package ai.djl.huggingface.tokenizers;

import ai.djl.training.util.DownloadUtils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;

public class BpeTokenizerBuilderTest {

    @Test
    public void testTokenizerCreation() throws IOException {
        Path bpe = Paths.get("build/BPE");
        Path vocab = bpe.resolve("vocab.json");
        Path merges = bpe.resolve("merges.txt");

        DownloadUtils.download(
                new URL("https://huggingface.co/microsoft/layoutlmv3-base/raw/main/vocab.json"),
                vocab,
                null);
        DownloadUtils.download(
                new URL("https://huggingface.co/microsoft/layoutlmv3-base/raw/main/merges.txt"),
                merges,
                null);

        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder().optTokenizerPath(bpe).build()) {
            Assert.assertNotNull(tokenizer);
        }
    }
}
