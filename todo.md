- After adding the forward loop, copying the weights from hf_model and generating a sample from the model.
- so we used GPT-2 model to initilaized our weights and got our prediction on the sample. 
**commit**
- random model initinialtion at random, from strach and then we want to train the model that would give us sequence as good 
as or better that we just got. 
- output from random model. 
> Hello, I'm a language model,IteratorWithinSax Gideon accredited references districts whats contestThough 216 BinaryfacJames Sicily underpin_-Pal SyracuseTER recover deleting
> Hello, I'm a language model,iji appearedlon Informationisine neighborhood enroltermin QuantumStephen Mayhem033 wormatche gobl Hubble sageuable Vengeanceす Bronze brushes
> Hello, I'm a language model,MC mocking psychedel unexpl routingULAR Pharma whats Championshipwaters greatestcolonial thoseString Enteredetti differ gravy Led Trust filtering bid
> Hello, I'm a language model, Evil upside compress piermet Goo strokes MIDI helldisciplinaryDoug alerts・Camolonprop loss closureplet vanquishedforkmult
> Hello, I'm a language model, paramsdocs cordsbizoding proudlyselves regex greenhouse Assadgewateralitiesurrencyanyahu substitute Tud explosion911Null nerdsHaving actress
**commit**
- added logic to load sample data, calculate the logits. 
**commit**
- added logic to calculate loss 
using device: cuda
Only loading first 1k charchters as model input
logits shape torch.Size([4, 32, 50257])
loss tensor(11.0371, device='cuda:0', grad_fn=<NllLossBackward0>)