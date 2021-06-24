const tf = require('@tensorflow/tfjs-node');

function normalized(data){ // i & r
    x = (data[0] * -3.013926423) + 17.32172345
    y = (data[1] * 0.503007298) + 24.97975683
    z = (data[2] * 14.70387591) + 20.3396711
    return [x, y, z]
}

function denormalized(data){
    x = (data[0] * 4.2064) + 17.32172345
    y = (data[1] * -2.565) + 24.97975683
    z = (data[2] * -289.072) + 20.3396711
    return [x, y, z]
}


async function predict(data){
    let in_dim = 3;
    
    data = normalized(data);
    shape = [1, in_dim];

    tf_data = tf.tensor2d(data, shape);

    try{
        // path load in public access => github
        const path = 'https://raw.githubusercontent.com/anzala0807/jst-delta-anzala-/main/public/ex_model/model.json';
        const model = await tf.loadGraphModel(path);
        
        predict = model.predict(
                tf_data
        );
        result = predict.dataSync();
        return denormalized( result );
        
    }catch(e){
      console.log(e);
    }
}

module.exports = {
    predict: predict 
}
  
