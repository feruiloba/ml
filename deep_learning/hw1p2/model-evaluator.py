import numpy as np
import torch
from tqdm import tqdm
from audio_dataset import AudioDataset, AudioTestDataset, config
from network import Network

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PHONEMES = [
            '[SIL]',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',
            'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
            'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
            'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
            'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
            'V',     'W',     'Y',     'Z',     'ZH',    '[SOS]', '[EOS]']
ROOT = "/home/feruiloba/ml/deep_learning/hw1p2/content/11785-f25-hw1p2" # Define the root directory of the dataset here

val_data = AudioDataset(root=ROOT, partition="dev-clean", phonemes=PHONEMES, context=config['context'])
val_loader = torch.utils.data.DataLoader(
    dataset     = val_data,
    num_workers = 0,
    batch_size  = 1,
    pin_memory  = True,
    drop_last   = True,
    shuffle     = False,
)

def test(model, test_loader):
    ### What you call for model to perform inference?
    model.eval() # TODO train or eval?

    ### List to store predicted phonemes of test data
    test_predictions =  []

    sample_size = 1000

    ### Which mode do you need to avoid gradients?
    with torch.no_grad():

        for i, mfccs in enumerate(tqdm(test_loader)):

            if i > sample_size:
                continue

            mfccs   = mfccs.to(device)

            logits  = model(mfccs)

            ### Get most likely predicted phoneme with argmax
            predicted_phoneme = torch.argmax(logits, dim=1)

            ### How do you store predicted_phonemes with test_predictions? HINT: look at the eval() function from earlier
            # Remember the phonemes were converted from strings to their corresponding integer indices earlier, and the results of the argmax is a list of the integer indices of the predicted phonemes.
            # So how do you get and store the actual predicted phonemes (strings NOT integers)
            # TODO: Convert predicted_phonemes (integer indices from argmax) back to phoneme strings and append them to test_predictions
            # raise NotImplementedError(
            #     "TODO: convert predicted_phonemes integer indices -> phoneme strings and append to test_predictions. "
            #     "Replace this exception with the correct code implementation."
            # )

            test_predictions.append(PHONEMES[predicted_phoneme])
    
    ## SANITY CHECK
    sample_predictions = test_predictions[:10]
    if not isinstance(sample_predictions[0], str):
        print(f"❌ ERROR: Predictions should be phoneme STRINGS, not {type(sample_predictions[0]).__name__}!")
        print(f"   You need to convert integer indices to their corresponding phoneme strings")
        print(f"   Hint: Look at the eval() function to get the idea")

    # Print a preview of predictions for manual inspection
    print("\nSample predictions:", sample_predictions)
    print("\nPredictions Generated successfully!")

    correct = 0
    real_values = []
    with torch.no_grad():
        for i, (mfccs, real_value) in enumerate(val_loader):

            if i > sample_size:
                continue

            real_value = real_value.to(device)

            real_values.append(PHONEMES[real_value])
            correct += test_predictions[i] == PHONEMES[real_value]

    print(f"Validation Accuracy: {correct/sample_size}")


    return (test_predictions, real_values)

# Generate model test predictions
INPUT_SIZE  = (2*config['context'] + 1) * 28
model       = Network(INPUT_SIZE, len(PHONEMES))

state_dict = torch.load('checkpoint_epoch_3_2025-09-17-01-01-56.pth', map_location=device)["model_state_dict"]
model.to(device)

# Load the state_dict into the model
model.load_state_dict(state_dict)


test_data = AudioTestDataset(root=ROOT, context=config['context'], partition="dev-clean")
test_loader = torch.utils.data.DataLoader(
    dataset     = test_data,
    num_workers = 0,
    batch_size  = 1,
    pin_memory  = True,
    drop_last   = False,
    shuffle     = False
)

(predictions, real_labels) = test(model, test_loader)

### Create CSV file with predictions

with open("./submission.csv", "w+") as f:
    f.write("id,label\n")
    for i in range(len(predictions)):
        f.write("{},{}\n".format(i, predictions[i]))

    print("submission.csv file created successfully!")

with open("./real.csv", "w+") as f:
    f.write("id,label\n")
    for i in range(len(real_labels)):
        f.write("{},{}\n".format(i, real_labels[i]))

    print("submission.csv file created successfully!")