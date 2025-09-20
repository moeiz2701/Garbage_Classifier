pipeline {
    agent any

    environment {
        DOCKERHUB_CREDENTIALS = credentials('dockerhub-credentials-id')  // Jenkins credentials ID
        DOCKER_IMAGE = "moeiz2701/garbage-classifier"
    }

    stages {
        stage('Checkout') {
            steps {
                deleteDir()
                git branch: 'master',
                    url: 'https://github.com/moeiz2701/Garbage_Classifier.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $DOCKER_IMAGE:$BUILD_NUMBER .'
            }
        }

        stage('Push to DockerHub') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-credentials-id', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    sh '''
                        echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin
                        docker push $DOCKER_IMAGE:$BUILD_NUMBER
                        docker tag $DOCKER_IMAGE:$BUILD_NUMBER $DOCKER_IMAGE:latest
                        docker push $DOCKER_IMAGE:latest
                    '''
                }
            }
        }

        stage('Notify Admin') {
            steps {
                mail to: 'admin@example.com',
                     subject: "Jenkins Job Success - Build #${BUILD_NUMBER}",
                     body: "The Jenkins pipeline completed successfully. Docker image pushed to DockerHub."
            }
        }
    }
}
