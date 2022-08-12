from django.shortcuts import render
import pickle
import numpy as np

# Create your views here.
def homePage(request):
    return render(request,'pages/index.html',{})

def analyseResult(request):
    if(request.method == 'POST'):
        age = request.POST.get('age')
        workclass = request.POST.get('workclass')
        fnlwgt = request.POST.get('fnlwgt')
        education = request.POST.get('education')
        educational_num = request.POST.get('educational_num')
        marital_status = request.POST.get('marital-status')
        occupation = request.POST.get('occupation')
        relationship = request.POST.get('relationship')
        race = request.POST.get('race')
        gender = request.POST.get('gender')
        capital_gain = request.POST.get('capital-gain')
        capital_loss = request.POST.get('capital-loss')
        hours_per_week = request.POST.get('hours-per-week')
        native_country = request.POST.get('native-country')

        

        model = pickle.load(open('model.pkl', 'rb'))

        prediction_data = [int(age), int(workclass), int(fnlwgt), int(education), int(educational_num), int(marital_status), int(
            occupation), int(relationship), int(race), int(gender), int(capital_gain), int(capital_loss), int(hours_per_week), int(native_country)]

        print(prediction_data)

        input_data_as_numpy_array = np.asarray(prediction_data)

        # reshape the np array as we are predicting for only one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        res = model.predict(input_data_reshaped)
        if(res[0] == 0):
            st = 'The person has less than 50,000 income'
            context = {'msg':st}
            return render(request,'pages/result.html',context)
        else:
            st = 'The person has an income more than of 50,000 income'
            context = {'msg':st}
            return render(request,'pages/result.html',context)

    else:
        return render(request,'pages/analyse.html',{})

def resultPage(request):
    return render(request,'pages/result.html')