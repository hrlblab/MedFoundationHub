#!/bin/bash
python app.py &
python ui.py
wait

#TIPS: How to use 'start.sh' in shell script:
#COPY start.sh .
#RUN chmod +x start.sh
#CMD ["./start.sh"]
