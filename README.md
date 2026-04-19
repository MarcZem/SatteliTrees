Erkennung von Entwaldung mit einem heybridem Ansatz aus U-Net und Random Forest	

Überblick  

In unserer Submission setzen wir auf eine Mischung aus verschiedenen Lernmethoden, um Flächen zu identifizieren, die abgeholzt wurden. Es wurde so entwickelt, dass es Störgeräusche minimiert und gleichzeitig die klaren Konturen der betroffenen Gebiete beibehält. Es nutzt Daten von verschiedenen Satelliten über mehrere Jahre hinweg, konkret von 2020 bis 2025.

Wie es funktioniert  
Der Algorithmus basiert auf einer speziellen Strategie mit zwei Hauptbestandteilen:

U-Net (Formgeber): Ein Modell, das darauf trainiert ist, feine Strukturen und Kanten zu erkennen. Es erstellt die erste Skizze der potenziellen Abholzungsgebiete.

Random Forest (Statistischer Prüfer): Dieses Modell analysiert über mehrere Jahre hinweg durchschnittliche Werte und Schwankungen der Spektraldaten sowie zusätzliche Merkmale. Es dient als Filter, um die Wahrscheinlichkeit einer echten Veränderung pro Pixel zu überprüfen.

Der Prozess mit der "Intelligenten Maske":  

In der Datenfusion werden Radar- und optische Daten kombiniert, zusammen mit tiefen Merkmalen. Das U-Net liefert die Form vor, aber das Random Forest-Modell muss diese durch eine erweiterte Maske bestätigen. Nur Flächen, die beide Anforderungen erfüllen, bleiben übrig im zu Nachberabeitung werden Morphologische Filter eingesetzt, um kleine Artefakte zu entfernen und so die Qualität der Daten für Anwendungen im Bereich Geoinformationssysteme zu verbessern. Während viele Modelle oft unklare Ränder zeigen, sorgt unser Ansatz dank des U-Net dafür, dass präzise Grenzen entstehen. Der Einsatz des Random Forest zur Validierung hilft dabei, saisonale Veränderungen oder Wolkenartefakte herauszufiltern, die andere Modelle oft in die Irre führen können. Außerdem schaut der Algorithmus nicht nur auf einzelne Zeitpunkte, sondern untersucht statistische Unterschiede über einen Zeitraum von sechs Jahren unter Berücksichtigung verschiedener Sensortypen.
